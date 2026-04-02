import os
import torch
import numpy as np

from datetime import datetime
from tqdm import tqdm
from time import time
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from wcode.training.loss.EntropyLoss import RobustCrossEntropyLoss
from wcode.training.Trainers.Weakly.Incomplete_Learning.AIO2.models import MeanTeacher
from wcode.training.Trainers.Weakly.Incomplete_Learning.AIO2.early_learning_detection import (
    ACT_module,
)
from wcode.training.Trainers.Weakly.Incomplete_Learning.AIO2.label_correction import (
    object_wise_label_correction,
)
from wcode.training.dataset.BaseDataset import BaseDataset
from wcode.training.loss.CompoundLoss import Tversky_and_CE_loss
from wcode.training.loss.DiceLoss import TverskyLoss
from wcode.training.logs_writer.logger_for_segmentation import logger
from wcode.training.learning_rate.PolyLRScheduler import PolyLRScheduler
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.file_operations import open_yaml, open_json, copy_file_to_dstFolder
from wcode.utils.others import empty_cache, dummy_context
from wcode.utils.collate_outputs import collate_outputs
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.NaturalImagePredictor import NaturalImagePredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)
from wcode.training.loss.EntropyLoss import MixCrossEntropyLoss


class AIO2Trainer(PatchBasedTrainer):
    def __init__(
        self,
        config_file_path: str,
        fold: int,
        alpha: float,
        wsize: list,
        filter_size: int,
        filter_all: bool,
        resume_flag: bool,
        dataset_type: str,
        awce_beta: float,
        improved_weight: float,
        implement_type: str,
        verbose: bool = False,
    ):
        # hyperparameter
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.wsize = wsize
        self.filter_size = filter_size
        self.filter_all = filter_all
        self.dataset_type = dataset_type
        self.resume_flag = resume_flag
        self.awce_beta = awce_beta
        self.improved_weight = improved_weight
        self.implement_type = implement_type
        assert self.implement_type in ["original", "improved"]
        assert self.dataset_type in ["Dense", "Sparse"]
        hyperparams_name = "alpha_{}_wsize_{}_filter_size_{}_filter_all_{}_dataset_type_{}_awce_beta_{}_improved_weight_{}_implement_type_{}".format(
            self.alpha,
            self.wsize,
            self.filter_size,
            self.filter_all,
            self.dataset_type,
            self.awce_beta,
            self.improved_weight,
            self.implement_type,
        )
        hyperparams_name = hyperparams_name.replace(" ", "")

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
        self.save_every = 5
        self.disable_checkpointing = False

        self.device = self.get_device()
        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None

        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

        if self.resume_flag:
            self.save_every = 1
            selected_I_r = os.path.join(self.logs_output_folder, "selected_I_r.pth")
            assert os.path.isfile(
                selected_I_r
            ), "File selected_I_r.pth is not found. You should set resume_flag to False to run stage-1 first."
            self.load_checkpoint(selected_I_r)

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

            self.network = self.get_networks(self.config_dict["Network"])
            if self.pretrained_weight is not None:
                self.print_to_log_file(
                    "Loading pretrained weight from {}".format(self.pretrained_weight)
                )
                load_pretrained_weights(self.network, self.pretrained_weight)
            self.network.to(self.device)

            self.print_to_log_file("Compiling network...")
            self.network = torch.compile(self.network)

            self.do_deep_supervision = self.config_dict["Network"]["deep_supervision"]
            self.optimizer, self.lr_scheduler = self.get_optimizers()

            # initialize the memorization of metrics
            self.IoU_on_noisy_label = []
            self.ngs_dict = None
            self.detect_eps = None

            self.Dice_and_CE_loss = Tversky_and_CE_loss(
                {
                    "batch_dice": True,
                    "alpha": 0.5,
                    "beta": 0.5,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                    "apply_nonlin": True,
                },
                {},
                weight_ce=1.0,
                weight_tversky=1.0,
                ignore_label=self.ignore_label,
            )
            self.tversky_loss = TverskyLoss(
                alpha=0.3,
                beta=0.7,
                smooth=1e-5,
                batch_dice=True,
                do_bg=False,
                ddp=self.is_ddp,
                apply_nonlin=True,
            )
            self.mixce_loss = MixCrossEntropyLoss(beta=self.awce_beta)

            self.val_loss = Tversky_and_CE_loss(
                {
                    "batch_dice": True,
                    "smooth": 1e-5,
                    "do_bg": True,
                    "ddp": self.is_ddp,
                    "apply_nonlin": True,
                },
                {},
                weight_ce=1,
                weight_tversky=1,
                ignore_label=self.ignore_label,
            )
            if self.do_deep_supervision:
                self.Dice_and_CE_loss = self._build_deep_supervision_loss_object(
                    self.Dice_and_CE_loss
                )
                self.tversky_loss = self._build_deep_supervision_loss_object(
                    self.tversky_loss
                )
                self.mixce_loss = self._build_deep_supervision_loss_object(
                    self.mixce_loss
                )
                self.val_loss = self._build_deep_supervision_loss_object(self.val_loss)

            self.was_initialized = True
        else:
            raise Exception("Initialization was done before initialize method???")

    def get_networks(self, network_settings):
        return MeanTeacher(network_settings, self.alpha)

    def get_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.student.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.base_lr, self.num_epochs)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, lr_scheduler

    def get_train_and_val_dataset(self):
        train_dataset = BaseDataset(
            self.dataset_name,
            self.preprocess_config,
            split="train",
            fold=self.fold,
            modality=self.modality,
        )

        val_dataset = BaseDataset(
            self.dataset_name,
            self.preprocess_config,
            split="val",
            fold=self.fold,
            modality=self.modality,
        )

        val_on_train_dataset = BaseDataset(
            self.dataset_name,
            self.preprocess_config,
            split="train",
            fold=self.fold,
            modality=self.modality,
        )

        self.val_for_train_iterations_per_epoch = (
            len(val_on_train_dataset) // self.batch_size
        )
        return train_dataset, val_dataset, val_on_train_dataset

    def get_train_and_val_dataloader(self):
        train_collator, val_collator = self.get_collator()
        train_dataset, val_dataset, val_for_train_dataset = (
            self.get_train_and_val_dataset()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=train_collator,
            drop_last=True,
            prefetch_factor=3,
        )
        # shuffle is True here, because we expect patch based validation to be more comprehensive.
        # If the number of validation iteration is smaller than the valset, the validation
        # throughout the entire training process cannot cover all the images in the valset.
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=max(1, self.num_processes // 2),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            collate_fn=val_collator,
            drop_last=True,
            prefetch_factor=2,
        )

        val_for_train_loader = DataLoader(
            val_for_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=max(1, self.num_processes // 2),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            collate_fn=val_collator,
            drop_last=True,
            prefetch_factor=2,
        )
        return train_loader, val_loader, val_for_train_loader

    def run_training(self):
        self.train_start()

        while self.current_epoch < self.num_epochs:
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

                if not self.resume_flag:
                    val_for_train_outputs = []
                    for batch_id in tqdm(
                        range(self.val_for_train_iterations_per_epoch),
                        disable=self.verbose,
                    ):
                        try:
                            val_for_train_outputs.append(
                                self.validation_step(next(self.iter_val_for_train))
                            )
                        except StopIteration:
                            self.iter_val_for_train = iter(
                                self.dataloader_val_for_train
                            )
                            val_for_train_outputs.append(
                                self.validation_step(next(self.iter_val_for_train))
                            )
                else:
                    val_for_train_outputs = None

                self.validation_epoch_end(val_outputs, val_for_train_outputs)
            self.epoch_end(self.current_epoch)
        self.train_end()

    def train_start(self):
        if not self.was_initialized:
            self.initialize()

        empty_cache(self.device)

        self.dataloader_train, self.dataloader_val, self.dataloader_val_for_train = (
            self.get_train_and_val_dataloader()
        )

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
            if self.resume_flag:
                self.save_checkpoint(
                    os.path.join(self.logs_output_folder, "checkpoint_latest.pth")
                )
            else:
                self.save_checkpoint(
                    os.path.join(
                        self.logs_output_folder,
                        "checkpoint_Epoch{}.pth".format(current_epoch + 1),
                    )
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

        I_r, self.ngs_dict, self.detect_eps = ACT_module(
            self.IoU_on_noisy_label, self.ngs_dict, self.wsize, self.detect_eps
        )
        if I_r > 0 and not self.resume_flag:
            # Reload the check point at I_r.
            ## find the checkpoint
            self.resume_flag = True
            epochs_find = int(np.round(I_r / self.save_every) * self.save_every)
            self.print_to_log_file(
                "Checkpoint at epoch {} is selected.".format(epochs_find)
            )
            os.rename(
                os.path.join(
                    self.logs_output_folder,
                    "checkpoint_Epoch{}.pth".format(epochs_find),
                ),
                os.path.join(self.logs_output_folder, "selected_I_r.pth"),
            )
            os.rename(
                os.path.join(self.logs_output_folder, "progress.png"),
                os.path.join(self.logs_output_folder, "progress_stage1.png"),
            )

            # del the saved checkpoint
            del_files = [
                i
                for i in os.listdir(self.logs_output_folder)
                if i.endswith(".pth") and "Epoch" in i
            ]
            for f in del_files:
                os.remove(os.path.join(self.logs_output_folder, f))

            # load selected checkpoint
            self.load_checkpoint(
                os.path.join(self.logs_output_folder, "selected_I_r.pth")
            )

            self.save_every = 1

    def train_epoch_start(self):
        self.iter_train = iter(self.dataloader_train)
        self.network.student.train()
        self.network.teacher.eval()
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
            outputs = self.network(images, train_flag=True)

            student_output, teacher_output = (
                outputs["student_out"]["pred"],
                outputs["teacher_out"]["pred"],
            )

            if self.implement_type == "original":
                # original implementation
                if self.resume_flag:
                    # performing label correction
                    labels_np = labels.detach().cpu().numpy()
                    teacher_hard_pred = torch.argmax(
                        teacher_output, dim=1, keepdim=True
                    )
                    teacher_hard_pred_np = (
                        teacher_hard_pred.to(torch.uint8).detach().cpu().numpy()
                    )
                    labels = object_wise_label_correction(
                        labels_np,
                        teacher_hard_pred_np,
                        self.device,
                        self.filter_size,
                        self.filter_all,
                        self.dataset_type,
                    )
                # Compute Loss
                l = self.Dice_and_CE_loss(student_output, labels)
            elif self.implement_type == "improved":
                # improved implementation
                if self.resume_flag:
                    # performing label correction
                    labels_np = labels.detach().cpu().numpy()
                    teacher_hard_pred = torch.argmax(
                        teacher_output, dim=1, keepdim=True
                    )
                    teacher_hard_pred_np = (
                        teacher_hard_pred.to(torch.uint8).detach().cpu().numpy()
                    )
                    labels_refined = object_wise_label_correction(
                        labels_np,
                        teacher_hard_pred_np,
                        self.device,
                        self.filter_size,
                        self.filter_all,
                        self.dataset_type,
                    )
                    # Compute Loss
                    l = self.Dice_and_CE_loss(
                        student_output, labels_refined
                    ) + self.improved_weight * (
                        self.tversky_loss(student_output, labels)
                        + self.mixce_loss(student_output, labels)
                    )
                else:
                    # Compute Loss
                    # l = self.Dice_and_CE_loss(student_output, labels)
                    l = self.tversky_loss(student_output, labels) + self.mixce_loss(
                        student_output, labels
                    )
            else:
                raise ValueError("Unsupport type of implementation.")

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

        # update teacher model using ema
        self.network.update_ema_variables()

        return {"loss": l.detach().cpu().numpy()}

    def train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)
        self.lr_scheduler.step()
        loss_here = np.mean(outputs["loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)

    def validation_epoch_start(self):
        self.iter_val = iter(self.dataloader_val)
        self.iter_val_for_train = iter(self.dataloader_val_for_train)
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

    def validation_epoch_end(self, val_outputs, val_for_train_outputs):
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

        if not self.resume_flag:
            # update metrics on noisy labels
            outputs_collated = collate_outputs(val_for_train_outputs)
            tp = np.sum(outputs_collated["tp_hard"], 0)
            fp = np.sum(outputs_collated["fp_hard"], 0)
            fn = np.sum(outputs_collated["fn_hard"], 0)

            global_iou_per_class = [
                i for i in [i / (i + j + k) for i, j, k in zip(tp, fp, fn)]
            ]
            mean_fg_iou = np.nanmean(global_iou_per_class)
            self.IoU_on_noisy_label.append(mean_fg_iou)
            self.print_to_log_file(
                "IoU on noisy label at stage-1",
                [np.round(i, decimals=4) for i in global_iou_per_class],
            )

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
