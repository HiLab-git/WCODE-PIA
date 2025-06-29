import os
import argparse

from wcode.training.Trainers.Weakly.NLL.MPNN.MPNNTrainer import MPNNTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting",
    type=str,
    default=None,
    help="File Name of Setting yaml, or you can just give the absolute path of the file.",
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--alpha",
    type=float,
    default=0.99,
    help="hyperparameter of EMA in MeanTeacher model",
)
parser.add_argument(
    "-K",
    type=int,
    default=5,
    help="number of linknet",
)
parser.add_argument(
    "-M",
    type=int,
    default=4,
    help="number of the Teacher's input",
)
parser.add_argument(
    "--consist_weight",
    type=float,
    default=0.1,
    help="weight of consistency learning",
)
parser.add_argument(
    "--consist_rampup",
    type=int,
    default=100,
    help="rampup epochs of consistency learning",
)
parser.add_argument(
    "--consist_epoch",
    type=int,
    default=50,
    help="epochs to start consistency learning",
)
parser.add_argument(
    "--num_linknets_epochs",
    type=int,
    default=300,
    help="epochs to train linknets",
)
parser.add_argument(
    "--train_linknets",
    action="store_true",
    help="Call this parameter (True) to train for Linknets, or (False) to train for MeanTeacher.",
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
    Trainer = MPNNTrainer(
        settings_path,
        args.f,
        alpha=args.alpha,
        K=args.K,
        M=args.M,
        consist_weight=args.consist_weight,
        consist_rampup=args.consist_rampup,
        consist_epoch=args.consist_epoch,
        num_linknets_epochs=args.num_linknets_epochs,
        train_linknets=args.train_linknets,
        improved_weight=args.improved_weight,
        implement_type=args.implement_type,
    )
    Trainer.run_training()
