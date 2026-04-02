import os
import argparse

from wcode.training.Trainers.Weakly.Incomplete_Learning.AIO2.AIO2Trainer import (
    AIO2Trainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--alpha",
    type=float,
    default=0.999,
    help="hyperparameter of EMA in MeanTeacher model",
)
parser.add_argument(
    "--wsize",
    type=int,
    nargs="+",
    default=[10, 20, 30, 40],
    help="List of sliding window sizes used for numerical gradient calculation",
)
parser.add_argument(
    "--filter_size",
    type=int,
    default=5,
    help="Size of convolutional filter to soften the label.",
)
parser.add_argument(
    "--filter_all",
    action="store_true",
    help="Soften the edge of all the instances (True, call this param) or newly detected instances (False, not call this param).",
)
parser.add_argument("--dataset_type", type=str, default="Dense", help="Dense or Sparse")
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
    Trainer = AIO2Trainer(
        settings_path,
        args.f,
        alpha=args.alpha,
        wsize=args.wsize,
        filter_size=args.filter_size,
        filter_all=args.filter_all,
        dataset_type=args.dataset_type,
        resume_flag=args.resume_train,
        awce_beta=args.AwCE_beta,
        improved_weight=args.improved_weight,
        implement_type=args.implement_type,
    )
    Trainer.run_training()
