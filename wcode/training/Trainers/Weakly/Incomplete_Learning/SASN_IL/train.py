import os
import argparse

from wcode.training.Trainers.Weakly.Incomplete_Learning.SASN_IL.SASN_ILTrainer import (
    SASN_ILTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--alpha",
    type=float,
    default=0.99,
    help="hyperparameter of EMA in MeanTeacher model",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.95,
    help="hyperparameter to define the end point of stage-1",
)
parser.add_argument(
    "--stage1_epochs",
    type=int,
    default=150,
    help="Training epochs of stage-1",
)
parser.add_argument(
    "--resume_train",
    action="store_true",
    help="Start training at stage-1 (False, not call this param) or stage-2 (True, call this param).",
)
parser.add_argument(
    "--AwCE_beta", type=float, default=1.0, help="beta of AwCE"
)
parser.add_argument(
    "--improved_weight",
    type=float,
    default=1.0,
    help="hyperparameter for improved version of loss.",
)
parser.add_argument(
    "--implement_type", type=str, default="original", help="can be original, improved"
)
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = SASN_ILTrainer(
        settings_path,
        args.f,
        alpha=args.alpha,
        tau=args.tau,
        stage1_epochs=args.stage1_epochs,
        resume_flag=args.resume_train,
        awce_beta=args.AwCE_beta,
        improved_weight=args.improved_weight,
        implement_type=args.implement_type,
    )
    # import torch
    # torch.autograd.set_detect_anomaly(True)
    Trainer.run_training()
