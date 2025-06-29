import os
import torch
import numpy as np

from datetime import datetime
from tqdm import tqdm
from time import time
from torch.amp import autocast
from torch.amp import GradScaler

from wcode.training.Trainers.Weakly.NLL.Coteaching.models import BiNet
from wcode.training.loss.EntropyLoss import RobustCrossEntropyLoss
from wcode.training.loss.EntropyLoss import MixCrossEntropyLoss
from wcode.training.loss.DiceLoss import TverskyLoss
from wcode.training.logs_writer.logger_for_segmentation import logger
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.file_operations import open_yaml, open_json, copy_file_to_dstFolder
from wcode.utils.others import empty_cache, dummy_context
from wcode.utils.collate_outputs import collate_outputs
from wcode.utils.ramps import ramps
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.NaturalImagePredictor import NaturalImagePredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)


class CoteachingTrainer(PatchBasedTrainer):
    """
    @article{han2018co,
        title={Co-teaching: Robust training of deep neural networks with extremely noisy labels},
        author={Han, Bo and Yao, Quanming and Yu, Xingrui and Niu, Gang and Xu, Miao and Hu, Weihua and Tsang, Ivor and Sugiyama, Masashi},
        journal={Advances in neural information processing systems},
        volume={31},
        year={2018}
    }
    """

    def __init__(
        self,
        config_file_path: str,
        fold: int,
        select_ratio: float,
        rampup_start: int,
        rampup_end: int,
        improved_weight: float,
        implement_type: str,
        verbose: bool = False,
    ):
        # hyperparameters
        self.select_ratio = select_ratio
        self.rampup_start = rampup_start
        self.rampup_end = rampup_end
        self.improved_weight = improved_weight
        self.implement_type = implement_type
        assert self.implement_type in ["original", "improved"]
        hyperparams_name = "select_ratio_{}_rampup_start_{}_rampup_end_{}_improved_weight_{}_implement_type_{}".format(
            self.select_ratio,
            self.rampup_start,
            self.rampup_end,
            self.improved_weight,
            self.implement_type,
        )

        self.verbose = verbose
        self.config_dict = open_yaml(config_file_path)
        if self.config_dict.__contains__("Inferring_settings"):
            del self.config_dict["Inferring_settings"]

        self.get_train_settings(self.config_dict["Training_settings"])
        self.fold = fold
        self.allow_mirroring_axes_during_inference = None

        self.was_initialized = False
        self._best_ema = None

        timestamp = datetime.now()
        time_ = "Train_Log_%d_%d_%d_%02.0d_%02.0d_%02.0d" % (
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )
        log_folder_name = self.method_name if self.method_name is not None else time_
        self.logs_output_folder = os.path.join(
            "./Logs",
            self.dataset_name,
            log_folder_name,
            hyperparams_name,
            "fold_" + self.fold,
        )
        config_and_code_save_path = os.path.join(
            self.logs_output_folder, "Config_and_code"
        )
        if not os.path.exists(config_and_code_save_path):
            os.makedirs(config_and_code_save_path)
        print("Training logs will be saved in:", self.logs_output_folder)

        # copy the config file to the logs folder
        copy_file_to_dstFolder(config_file_path, config_and_code_save_path)

        # copy the trainer file to the logs folder
        script_path = os.path.abspath(__file__)
        copy_file_to_dstFolder(script_path, config_and_code_save_path)

        self.log_file = os.path.join(self.logs_output_folder, time_ + ".txt")
        self.logger = logger()

        self.current_epoch = 0

        # checkpoint saving stuff
        self.save_every = 1
        self.disable_checkpointing = False

        self.device = self.get_device()
        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None

        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

    def initialize(self):
        if not self.was_initialized:
            self.is_ddp = False
            self.init_random()
            self.setting_check()

            if len(self.config_dict["Network"]["kernel_size"][0]) == 3:
                self.preprocess_config = "3d"
            elif len(self.config_dict["Network"]["kernel_size"][0]) == 2:
                self.preprocess_config = "2d"
            else:
                raise Exception()

            # we do not do deep supervision because the pce can be nan in lower resolutions.
            self.do_deep_supervision = False
            self.config_dict["Network"]["deep_supervision"] = False
            self.network = self.get_networks(self.config_dict["Network"]).to(
                self.device
            )
            self.print_to_log_file("Compiling network...")
            self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.get_optimizers()

            # rampup the weight of the loss of pseudo label
            self.ramp = ramps(
                start_iter=self.rampup_start,
                end_iter=self.rampup_end,
                start_value=0,
                end_value=1,
                mode="linear",
            )

            # train loss
            self.ce_loss = RobustCrossEntropyLoss(reduction="none")
            self.tversky_loss = TverskyLoss(
                alpha=0.3,
                beta=0.7,
                smooth=1e-5,
                batch_dice=True,
                do_bg=False,
                ddp=self.is_ddp,
                apply_nonlin=True,
            )
            self.mixce_loss = MixCrossEntropyLoss()

            # validation loss
            self.val_loss = RobustCrossEntropyLoss()

            if self.do_deep_supervision:
                self.ce_loss = self._build_deep_supervision_loss_object(self.ce_loss)
                self.tversky_loss = self._build_deep_supervision_loss_object(self.tversky_loss)
                self.mixce_loss = self._build_deep_supervision_loss_object(self.mixce_loss)
                self.val_loss = self._build_deep_supervision_loss_object(self.val_loss)

            self.was_initialized = True
        else:
            raise Exception("Initialization was done before initialize method???")

    def get_networks(self, network_settings):
        return BiNet(network_settings)

    def run_training(self):
        self.train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.epoch_start()

            self.train_epoch_start()
            train_outputs = []
            for batch_id in tqdm(
                range(self.tr_iterations_per_epoch), disable=self.verbose
            ):
                try:
                    train_outputs.append(self.train_step(next(self.iter_train)))
                except StopIteration:
                    self.iter_train = iter(self.dataloader_train)
                    train_outputs.append(self.train_step(next(self.iter_train)))
            self.train_epoch_end(train_outputs)

            with torch.no_grad():
                self.validation_epoch_start()
                val_outputs = []
                for batch_id in tqdm(
                    range(self.val_iterations_per_epoch), disable=self.verbose
                ):
                    try:
                        val_outputs.append(self.validation_step(next(self.iter_val)))
                    except StopIteration:
                        self.iter_val = iter(self.dataloader_val)
                        val_outputs.append(self.validation_step(next(self.iter_val)))
                self.validation_epoch_end(val_outputs)
            self.epoch_end(epoch)
        self.train_end()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val = self.get_train_and_val_dataloader()

    def train_end(self):
        self.save_checkpoint(
            os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        )

        if os.path.isfile(
            os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
        ):
            os.remove(os.path.join(self.logs_output_folder, "checkpoint_latest.pth"))

        empty_cache(self.device)
        self.print_to_log_file("Training done.")
        self.perform_actual_validation(save_probabilities=False)

    def epoch_start(self):
        self.logger.log("epoch_start_timestamps", time(), self.current_epoch)

    def epoch_end(self, epoch):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss", np.round(self.logger.logging["train_losses"][-1], decimals=4)
        )
        self.print_to_log_file(
            "val_loss", np.round(self.logger.logging["val_losses"][-1], decimals=4)
        )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.logging["dice_per_class"][-1]
            ],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.logging['epoch_end_timestamps'][-1] - self.logger.logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
            )

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(
                os.path.join(self.logs_output_folder, "checkpoint_best.pth")
            )

        self.logger.plot_progress_png(self.logs_output_folder)

        self.current_epoch += 1

    def train_epoch_start(self):
        self.iter_train = iter(self.dataloader_train)
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        # self.lr_scheduler.step()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log(
            "learning_rates", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )

    def train_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        # to device
        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
            labels = labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            net1_logits, net2_logits = (
                outputs["net1_out"]["pred"],
                outputs["net2_out"]["pred"],
            )
            # # compute loss for label (improved)
            # net1_loss = self.ce_loss(net1_logits, labels)
            # noisy_net1_loss = net1_loss[labels[:, 0] == 0].flatten()
            # noisy_net1_loss_sorted_idx = noisy_net1_loss.argsort()

            # net2_loss = self.ce_loss(net2_logits, labels)
            # noisy_net2_loss = net2_loss[labels[:, 0] == 0].flatten()
            # noisy_net2_loss_sorted_idx = noisy_net2_loss.argsort()

            # forget_ratio = (1 - self.select_ratio) * self.ramp.get_value(
            #     self.current_epoch
            # )
            # num_exchange = int((1 - forget_ratio) * len(noisy_net1_loss_sorted_idx))

            # # Compute Loss
            # l_noisy = (
            #     noisy_net1_loss[noisy_net2_loss_sorted_idx[:num_exchange]].mean()
            #     + noisy_net2_loss[noisy_net1_loss_sorted_idx[:num_exchange]].mean()
            # )
            # l_fg = 0
            # mask = labels[:, 0] != 0
            # if mask.any():
            #     l_fg += net1_loss[mask].mean()
            #     l_fg += net2_loss[mask].mean()
            # l = l_noisy + l_fg
            # compute loss for label
            net1_loss = self.ce_loss(net1_logits, labels).flatten()
            noisy_net1_loss_sorted_idx = net1_loss.argsort()

            net2_loss = self.ce_loss(net2_logits, labels).flatten()
            noisy_net2_loss_sorted_idx = net2_loss.argsort()

            forget_ratio = (1 - self.select_ratio) * self.ramp.get_value(
                self.current_epoch
            )

            num_exchange = int((1 - forget_ratio) * len(noisy_net1_loss_sorted_idx))

            # Compute Loss
            l = (
                net1_loss[noisy_net2_loss_sorted_idx[:num_exchange]].mean()
                + net2_loss[noisy_net1_loss_sorted_idx[:num_exchange]].mean()
            )

            if self.implement_type == "improved":
                l += self.improved_weight * (
                    self.tversky_loss(net1_logits, labels)
                    + self.mixce_loss(net1_logits, labels)
                    + self.tversky_loss(net2_logits, labels)
                    + self.mixce_loss(net2_logits, labels)
                )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)
        self.lr_scheduler.step()
        loss_here = np.mean(outputs["loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)

    def validation_epoch_start(self):
        self.iter_val = iter(self.dataloader_val)
        self.network.eval()

    def validation_step(self, batch):
        # images in (b, c, (z,) y, x) and labels in (b, 1, (z,) y, x) or list object if do deep supervision
        images = batch["image"]
        labels = batch["label"]

        images = images.to(self.device, non_blocking=True)
        if isinstance(labels, list):
            labels = [i.to(self.device, non_blocking=True) for i in labels]
        else:
            labels = labels.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            if isinstance(outputs, dict):
                outputs = outputs["pred"]
            del images
            l = self.val_loss(outputs, labels)

        # use the new name of outputs and labels, so that you only need to change the network inference process
        # during validation and the variable name assignment code below, without changing any evaluation code.
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
            target = labels[0]
        else:
            output = outputs
            target = labels

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=None
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def validation_epoch_end(self, val_outputs):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("----------Perform actual validation----------")
        dataset_path = os.path.join("./Dataset_preprocessed", self.dataset_name)
        original_img_folder = os.path.join(
            dataset_path, "preprocessed_datas_" + self.preprocess_config
        )

        predictions_save_folder = os.path.join(self.logs_output_folder, "validation")
        if predictions_save_folder is not None and not os.path.exists(
            predictions_save_folder
        ):
            os.makedirs(predictions_save_folder)

        model_path = os.path.join(self.logs_output_folder, "checkpoint_final.pth")
        self.network = self.get_networks(self.config_dict["Network"])
        load_pretrained_weights(self.network, model_path, load_all=True, verbose=True)
        self.network.to(self.device)
        self.print_to_log_file("Compiling network for actual validation...")
        self.network = torch.compile(self.network)

        best_saved_model = torch.load(
            os.path.join(self.logs_output_folder, "checkpoint_best.pth"),
            weights_only=False,
        )
        self.print_to_log_file("Pseudo Best Epoch:", best_saved_model["current_epoch"])
        del best_saved_model

        predict_configs = {
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "fold": self.fold,
            "split": "val",
            "original_img_folder": original_img_folder,
            "predictions_save_folder": predictions_save_folder,
            "model_path": model_path,
            "device": self.device_dict,
            "overwrite": True,
            "save_probabilities": save_probabilities,
            "patch_size": self.patch_size,
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "perform_everything_on_gpu": True,
            "use_mirroring": True,
            "allowed_mirroring_axes": self.allow_mirroring_axes_during_inference,
            "num_processes": self.num_processes,
        }
        self.config_dict["Inferring_settings"] = predict_configs

        dataset_split = open_json(
            os.path.join(
                "./Dataset_preprocessed", self.dataset_name, "dataset_split.json"
            )
        )

        data_path_list = [
            i
            for i in os.listdir(original_img_folder)
            if i.endswith(".npy") and not i.endswith("_seg.npy")
        ]
        validation_data_file = [
            i
            for i in data_path_list
            if i.split(".")[0]
            in dataset_split["0" if self.fold == "all" else self.fold]["val"]
        ]
        validation_data_file.sort()
        validation_data_path = [
            os.path.join(original_img_folder, i) for i in validation_data_file
        ]
        validation_pkl_path = [
            os.path.join(original_img_folder, i.replace(".npy", ".pkl"))
            for i in validation_data_file
        ]
        predictions_save_path = [
            os.path.join(predictions_save_folder, i.replace(".npy", ""))
            for i in validation_data_file
        ]

        iter_lst = []
        for data, output_file, data_properites in zip(
            validation_data_path, predictions_save_path, validation_pkl_path
        ):
            iter_lst.append(
                {
                    "data": data,
                    "output_file": output_file,
                    "data_properites": data_properites,
                }
            )

        if self.natural_image_flag:
            predictor = NaturalImagePredictor(
                self.config_dict, allow_tqdm=True, verbose=False
            )
            predictor.manual_initialize(
                self.network, self.config_dict["Network"]["out_channels"]
            )
            self.print_to_log_file("Start predicting using NaturalImagePredictor.")
            start = time()
            predictor.predict_from_data_iterator(
                data_iterator=iter_lst,
                save_vis_mask=True,
                save_or_return_probabilities=save_probabilities,
            )
            self.print_to_log_file("Predicting ends. Cost: {}s".format(time() - start))
        else:
            predictor = PatchBasedPredictor(
                self.config_dict, allow_tqdm=True, verbose=False
            )
            predictor.manual_initialize(
                self.network, self.config_dict["Network"]["out_channels"]
            )
            self.print_to_log_file("Start predicting using PatchBasedPredictor.")
            start = time()
            predictor.predict_from_data_iterator(
                data_iterator=iter_lst,
                predict_way=self.preprocess_config,
                save_or_return_probabilities=save_probabilities,
            )
            self.print_to_log_file("Predicting ends. Cost: {}s".format(time() - start))

        ground_truth_folder = os.path.join(dataset_path, "gt_segmentations")
        evaluator = Evaluator(
            predictions_save_folder,
            ground_truth_folder,
            dataset_yaml_or_its_path=self.dataset_yaml,
            num_processes=self.num_processes,
        )
        evaluator.run()
        self.print_to_log_file("Evaluating ends.")
