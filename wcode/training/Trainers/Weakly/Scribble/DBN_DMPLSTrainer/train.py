import os
import argparse

from wcode.training.Trainers.Weakly.Scribble.DBN_DMPLSTrainer.DBN_DMPLSTrainer import (
    DBN_DMPLSTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--lambda_for_plabel",
    type=float,
    default=0.5,
    help="(hyperparameter) the weight of pseudo label",
)
parser.add_argument(
    "--rampup_epoch",
    type=int,
    default=100,
    help="(hyperparameter) the number of rampup epochs",
)
parser.add_argument(
    "--w_class",
    nargs="+",
    type=float,
    default=None,
    help="weight of class in partial CrossEntropyLoss",
)
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = DBN_DMPLSTrainer(
        settings_path,
        args.f,
        args.lambda_for_plabel,
        args.rampup_epoch,
        args.w_class,
    )
    Trainer.run_training()
