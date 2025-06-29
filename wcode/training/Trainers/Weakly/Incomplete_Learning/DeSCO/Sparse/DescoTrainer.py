import os
import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from datetime import datetime
from tqdm import tqdm
from time import time, sleep
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch._dynamo import OptimizedModule
from typing import Tuple, Union, List

from wcode.training.data_augmentation.transformation_list import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    SpatialTransform,
    RandomTransform,
    GaussianNoiseTransform,
    GaussianBlurTransform,
    MultiplicativeBrightnessTransform,
    ContrastTransform,
    BGContrast,
    SimulateLowResolutionTransform,
    GammaTransform,
    MirrorTransform,
    LabelValueTransform,
    DownsampleSegForDSTransform,
    ComposeTransforms,
    BasicTransform,
)
from wcode.training.Trainers.Weakly.Incomplete_Learning.DeSCO.Sparse.Collater import (
    LabelWeightedCollater,
)
from wcode.training.data_augmentation.custom_transforms.scalar_type import RandomScalar
from wcode.training.data_augmentation.compute_initial_patch_size import get_patch_size
from wcode.preprocessing.resampling import ANISO_THRESHOLD
from wcode.training.Trainers.Weakly.Incomplete_Learning.DeSCO.models import BiNet
from wcode.training.dataset.CrossDataset import CrossDataset
from wcode.training.dataset.BaseDataset import BaseDataset
from wcode.training.loss.CompoundLoss import Tversky_and_CE_loss
from wcode.training.loss.DiceLoss import TverskyLoss
from wcode.training.loss.EntropyLoss import RobustCrossEntropyLoss
from wcode.training.loss.deep_supervision import DeepSupervisionWeightedSummator
from wcode.training.logs_writer.logger_for_segmentation import logger
from wcode.training.dataloader.Collater import PatchBasedCollater
from wcode.training.learning_rate.PolyLRScheduler import PolyLRScheduler
from wcode.training.metrics import get_tp_fp_fn_tn
from wcode.utils.file_operations import open_yaml, open_json, copy_file_to_dstFolder
from wcode.utils.others import empty_cache, dummy_context
from wcode.utils.collate_outputs import collate_outputs
from wcode.utils.data_io import files_ending_for_2d_img, files_ending_for_sitk
from wcode.utils.ramps import ramps
from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.NaturalImagePredictor import NaturalImagePredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.inferring.utils.load_pretrain_weight import load_pretrained_weights
from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)


class DescoTrainer(PatchBasedTrainer):
    """
    @inproceedings{cai2023orthogonal,
        title={Orthogonal annotation benefits barely-supervised medical image segmentation},
        author={Cai, Heng and Li, Shumeng and Qi, Lei and Yu, Qian and Shi, Yinghuan and Gao, Yang},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3302--3311},
        year={2023}
    }
    """

    def __init__(
        self,
        config_file_path: str,
        fold: int,
        slice_weight: float,
        dropout_n: int,
        consistency_weight: float,
        threshold: float,
        rampup_epoch: int,
        verbose: bool = False,
    ):
        # hyperparameters
        self.dropout_n = dropout_n
        self.consistency_weight = consistency_weight
        self.threshold = threshold
        self.rampup_epoch = rampup_epoch
        self.slice_weight = slice_weight
        hyperparams_name = "dropout_n_{}_consistency_weight_{}_threshold_{}_rampup_epoch_{}_slice_weight_{}".format(
            self.dropout_n,
            self.consistency_weight,
            self.threshold,
            self.rampup_epoch,
            self.slice_weight,
        )

        self.verbose = verbose
        self.config_dict = open_yaml(config_file_path)
        if self.config_dict.__contains__("Inferring_settings"):
            del self.config_dict["Inferring_settings"]

        self.get_train_settings(self.config_dict["Training_settings"])
        self.fold = fold
        self.class_num = self.config_dict["Network"]["out_channels"]

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

    def get_train_settings(self, training_setting_dict):
        self.dataset_dict = training_setting_dict["dataset_dict"]
        self.dataset_name = self.dataset_dict["image"][0]
        self.dataset_yaml = open_yaml(
            os.path.join("./Dataset_preprocessed", self.dataset_name, "dataset.yaml")
        )

        if self.dataset_yaml["files_ending"] in files_ending_for_sitk:
            self.natural_image_flag = False
        elif self.dataset_yaml["files_ending"] in files_ending_for_2d_img:
            self.natural_image_flag = True
        else:
            raise ValueError("Not supporting file extension.")

        self.modality = training_setting_dict["modality"]
        if self.modality == None or self.modality == "all":
            self.modality = [
                int(i) for i in range(len(self.dataset_yaml["channel_names"]))
            ]
        self.channel_names = np.array(
            [i for i in self.dataset_yaml["channel_names"].values()]
        )[self.modality]
        self.method_name = training_setting_dict["method_name"]
        self.device_dict = training_setting_dict["device"]
        self.num_epochs = training_setting_dict["epoch"]
        self.tr_iterations_per_epoch = training_setting_dict["tr_iterations_per_epoch"]
        self.val_iterations_per_epoch = training_setting_dict[
            "val_iterations_per_epoch"
        ]
        self.batch_size = training_setting_dict["batch_size"]
        self.patch_size = training_setting_dict["patch_size"]
        self.base_lr = training_setting_dict["base_lr"]
        self.weight_decay = training_setting_dict["weight_decay"]
        self.num_processes = training_setting_dict["num_processes"]
        self.deterministic = training_setting_dict["deterministic"]
        self.random_seed = training_setting_dict["seed"]
        self.oversample_rate = training_setting_dict["oversample_rate"]
        self.probabilistic_oversampling = training_setting_dict[
            "probabilistic_oversampling"
        ]
        self.ignore_label = training_setting_dict["ignore_label"]
        self.checkpoint_path = training_setting_dict["checkpoint"]
        self.pretrained_weight = training_setting_dict["pretrained_weight"]

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

            (
                self.optimizer_net1,
                self.optimizer_net2,
                self.lr_scheduler_net1,
                self.lr_scheduler_net2,
            ) = self.get_optimizers()

            # rampup the weight of the loss of CPS
            self.ramp = ramps(
                start_iter=0,
                end_iter=self.rampup_epoch - 1,
                start_value=0,
                end_value=self.consistency_weight,
                mode="sigmoid",
            )

            # train loss
            self.dc_loss = TverskyLoss(
                alpha=0.5,
                beta=0.5,
                batch_dice=True,
                do_bg=False,
                ddp=self.is_ddp,
                apply_nonlin=True,
            )
            self.ce_loss = RobustCrossEntropyLoss(reduction="none")

            # validation loss
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
                self.dc_loss = self._build_deep_supervision_loss_object(self.dc_loss)
                self.ce_loss = self._build_deep_supervision_loss_object(self.ce_loss)
                self.val_loss = self._build_deep_supervision_loss_object(self.val_loss)

            self.was_initialized = True
        else:
            raise Exception("Initialization was done before initialize method???")

    def get_networks(self, network_settings):
        return BiNet(network_settings)

    def get_optimizers(self):
        optimizer_net1 = torch.optim.SGD(
            self.network.net1.parameters(),
            self.base_lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        optimizer_net2 = torch.optim.SGD(
            self.network.net2.parameters(),
            self.base_lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        lr_scheduler_net1 = PolyLRScheduler(
            optimizer_net1, self.base_lr, self.num_epochs
        )
        lr_scheduler_net2 = PolyLRScheduler(
            optimizer_net2, self.base_lr, self.num_epochs
        )
        
        return optimizer_net1, optimizer_net2, lr_scheduler_net1, lr_scheduler_net2

    def get_train_and_val_dataset(self):
        train_dataset = CrossDataset(
            self.dataset_dict,
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
        return train_dataset, val_dataset

    def get_collator(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        deep_supervision_scales = self._get_deep_supervision_scales()

        train_transform = self.get_training_transforms(
            self.patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
        )
        val_transform = self.get_validation_transforms(
            self.patch_size, deep_supervision_scales
        )

        train_collator = LabelWeightedCollater(
            self.slice_weight,
            self.batch_size,
            initial_patch_size,
            self.patch_size,
            self.oversample_rate,
            self.probabilistic_oversampling,
            train_transform,
        )
        val_collator = PatchBasedCollater(
            self.batch_size,
            self.patch_size,
            self.patch_size,
            self.oversample_rate,
            self.probabilistic_oversampling,
            val_transform,
        )
        return train_collator, val_collator

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

        (
            reg_coronal_label,
            reg_trans_label,
            coronal_slice_w,
            trans_slice_w,
            coronal_slice,
            trans_slice,
        ) = labels.permute(1, 0, 2, 3, 4)

        self.optimizer_net1.zero_grad(set_to_none=True)
        self.optimizer_net2.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            outputs = self.network(images)
            logits1, logits2 = (
                outputs["net1_out"]["pred"],
                outputs["net2_out"]["pred"],
            )

            # compute loss for label
            sup_loss = 0.5 * (
                self.dc_loss(logits1, reg_coronal_label, coronal_slice_w)
                + (self.ce_loss(logits1, reg_coronal_label) * coronal_slice_w).sum()
                / coronal_slice_w.sum()
                + self.dc_loss(logits2, reg_trans_label, trans_slice_w)
                + (self.ce_loss(logits2, reg_trans_label) * trans_slice_w).sum()
                / trans_slice_w.sum()
            )

            def entropy(prob):
                return -1.0 * torch.sum(prob * torch.log(prob + 1e-6), dim=1)

            # estimate uncertainty
            shape = [self.dropout_n] + list(logits1.shape)
            p1 = torch.zeros(shape, device=self.device)
            p2 = torch.zeros(shape, device=self.device)

            for i in range(self.dropout_n):
                noise_input = images + torch.clamp(
                    torch.randn_like(images) * 0.1, -0.2, 0.2
                )
                with torch.no_grad():
                    o = self.network(noise_input)

                o1, o2 = o["net1_out"]["pred"], o["net2_out"]["pred"]
                p1[i] = o1.reshape(logits1.shape)
                p2[i] = o2.reshape(logits2.shape)

            p1, p2 = p1.softmax(dim=2).mean(0), p2.softmax(dim=2).mean(0)
            u1, u2 = entropy(p1), entropy(p2)
            # pseudolabel1, pseudolabel2 = logits1.argmax(1),logits2.argmax(1)

            alpha = self.ramp.get_value(self.current_epoch)
            threshold = (self.threshold + (1 - self.threshold) * alpha) * np.log(2)

            mask1 = (u1 < threshold).float()
            mask2 = (u2 < threshold).float()

            cps_loss = 0.5 * (
                (
                    self.cps_loss(logits1, logits2.argmax(dim=1, keepdim=True)) * mask2
                ).sum()
                / mask2.sum()
                + (
                    self.cps_loss(logits2, logits1.argmax(dim=1, keepdim=True)) * mask1
                ).sum()
                / mask1.sum()
            )
            
            # Compute Loss
            l = (1 - alpha) * sup_loss + alpha * cps_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer_net1)
            self.grad_scaler.unscale_(self.optimizer_net2)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer_net1)
            self.grad_scaler.step(self.optimizer_net2)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer_net1.step()
            self.optimizer_net2.step()

        return {"loss": l.detach().cpu().numpy()}

    def train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)
        self.lr_scheduler_net1.step()
        self.lr_scheduler_net2.step()
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

    def save_checkpoint(self, filename: str) -> None:
        if not self.disable_checkpointing:
            mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            checkpoint = {
                "network_weights": mod.state_dict(),
                "optimizer_state": [
                    self.optimizer_net1.state_dict(),
                    self.optimizer_net2.state_dict(),
                ],
                "grad_scaler_state": (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None
                ),
                "logging": self.logger.get_checkpoint(),
                "_best_ema": self._best_ema,
                "current_epoch": self.current_epoch + 1,
                "LRScheduler_step": [
                    self.lr_scheduler_net1.ctr,
                    self.lr_scheduler_net2.ctr,
                ],
            }
            torch.save(checkpoint, filename)
        else:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")

    def load_checkpoint(self, filename_or_checkpoint):
        self.print_to_log_file("Load checkpoint...")
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith(
                "module."
            ):
                key = key[7:]
            new_state_dict[key] = value

        if "checkpoint_final.pth" in os.listdir(self.logs_output_folder):
            self.current_epoch = self.num_epochs
        else:
            self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]

        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)

        self.optimizer_net1.load_state_dict(checkpoint["optimizer_state"][0])
        self.optimizer_net2.load_state_dict(checkpoint["optimizer_state"][1])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.lr_scheduler_net1.ctr = checkpoint["LRScheduler_step"][0]
        self.lr_scheduler_net2.ctr = checkpoint["LRScheduler_step"][1]


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
            os.path.join(self.logs_output_folder, "checkpoint_best.pth"), weights_only=False
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
