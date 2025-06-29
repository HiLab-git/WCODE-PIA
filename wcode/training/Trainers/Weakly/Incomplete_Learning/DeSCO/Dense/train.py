import os
import argparse

from wcode.training.Trainers.Weakly.Incomplete_Learning.DeSCO.Dense.DenseDeSCOTrainer import (
    DenseDeSCOTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")

parser.add_argument(
    "--dropout_n", type=int, default=4, help="inferring times to get uncertainty map."
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
    "--rampup_epoch",
    type=int,
    default=50,
    help="(hyperparameter) the number of rampup epochs",
)
parser.add_argument(
    "--implement_type", type=str, default="original", help="can be original, improved"
)
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = DenseDeSCOTrainer(
        settings_path,
        args.f,
        dropout_n=args.dropout_n,
        consistency_weight=args.consistency_weight,
        threshold=args.threshold,
        rampup_epoch=args.rampup_epoch,
        implement_type=args.implement_type,
    )
    Trainer.run_training()
