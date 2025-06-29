import os
import argparse

from wcode.training.Trainers.Weakly.NLL.Coteaching.CoteachingTrainer import (
    CoteachingTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")

parser.add_argument(
    "--select_ratio", type=float, default=0.999, help="hyperparameter in NLL"
)
parser.add_argument(
    "--rampup_start",
    type=int,
    default=25,
    help="(hyperparameter) the number of rampup epochs",
)
parser.add_argument(
    "--rampup_end",
    type=int,
    default=75,
    help="(hyperparameter) the number of rampup epochs",
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
    Trainer = CoteachingTrainer(
        settings_path,
        args.f,
        select_ratio=args.select_ratio,
        rampup_start=args.rampup_start,
        rampup_end=args.rampup_end,
        improved_weight=args.improved_weight,
        implement_type=args.implement_type,
    )
    Trainer.run_training()
