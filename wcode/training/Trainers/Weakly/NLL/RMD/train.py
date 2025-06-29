import os
import argparse

from wcode.training.Trainers.Weakly.NLL.RMD.RMDTrainer import RMDTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")

parser.add_argument(
    "--select_ratio", type=float, default=0.999, help="(hyperparameter) in Co-teaching."
)
parser.add_argument(
    "--rampup_start_CoT",
    type=int,
    default=25,
    help="(hyperparameter) the number of rampup epochs for Co-teaching.",
)
parser.add_argument(
    "--rampup_end_CoT",
    type=int,
    default=75,
    help="(hyperparameter) the number of rampup epochs for Co-teaching.",
)

parser.add_argument(
    "--consistency_weight",
    type=float,
    default=1,
    help="(hyperparameter) in consistency learning.",
)
parser.add_argument(
    "--rampup_start_Con",
    type=int,
    default=25,
    help="(hyperparameter) the number of rampup epochs for Consistency.",
)
parser.add_argument(
    "--rampup_end_Con",
    type=int,
    default=75,
    help="(hyperparameter) the number of rampup epochs for Consistency.",
)

parser.add_argument(
    "--num_aug",
    type=int,
    default=1,
    help="(hyperparameter) times of doing stong augmentation per iteration.",
)
parser.add_argument(
    "--strong_aug_start_epoch",
    type=int,
    default=75,
    help="(hyperparameter) the start epoch of strong augmentation.",
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
    Trainer = RMDTrainer(
        settings_path,
        args.f,
        select_ratio=args.select_ratio,
        rampup_start_CoT=args.rampup_start_CoT,
        rampup_end_CoT=args.rampup_end_CoT,
        consistency_weight=args.consistency_weight,
        rampup_start_Con=args.rampup_start_Con,
        rampup_end_Con=args.rampup_end_Con,
        num_aug=args.num_aug,
        strong_aug_start_epoch=args.strong_aug_start_epoch,
        improved_weight=args.improved_weight,
        implement_type=args.implement_type,
    )
    Trainer.run_training()
