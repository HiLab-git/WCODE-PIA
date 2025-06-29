import os
import argparse

from wcode.training.Trainers.Weakly.Partial_Instance.Desco.DescoTrainer import (
    DescoTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--slice_weight", type=float, default=0.95, help="hyperparameter in Tversky loss"
)
parser.add_argument(
    "--dropout_n", type=int, default=8, help="hyperparameter in Tversky loss"
)
parser.add_argument(
    "--consistency_weight",
    type=float,
    default=0.1,
    help="hyperparameter in Tversky loss",
)
parser.add_argument(
    "--threshold", type=float, default=0.75, help="hyperparameter in Tversky loss"
)
parser.add_argument(
    "--sce_alpha", type=float, default=0.8, help="hyperparameter in SCE loss"
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.3,
    help="sharpen the soft label",
)
parser.add_argument(
    "--tversky_alpha", type=float, default=0.4, help="hyperparameter in Tversky loss"
)
parser.add_argument(
    "--lambda_for_plabel",
    type=float,
    default=1.0,
    help="(hyperparameter) the final value of rampup",
)
parser.add_argument(
    "--rampup_epoch",
    type=int,
    default=100,
    help="(hyperparameter) the number of rampup epochs",
)
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = DescoTrainer(
        settings_path,
        args.slice_weight,
        args.dropout_n,
        args.consistency_weight,
        args.threshold,
        args.f,
        args.tversky_alpha,
        args.sce_alpha,
        args.tau,
        args.lambda_for_plabel,
        args.rampup_epoch,
    )
    Trainer.run_training()
