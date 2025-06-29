import os
import argparse

from wcode.training.Trainers.Weakly.Incomplete_Learning.ReCo_I2P.ReCo_I2PTrainer import (
    ReCo_I2PTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--tversky_alpha", type=float, default=0.3, help="hyperparameter in Tversky loss"
)
parser.add_argument(
    "--num_prototype",
    type=int,
    default=3,
    help="hyperparameter: memoried prototype",
)
parser.add_argument(
    "--memory_rate",
    type=float,
    default=0.99,
    help="hyperparameter: updating rate of memoried prototype.",
)
parser.add_argument(
    "--lambda_for_C",
    type=float,
    default=0.1,
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
    Trainer = ReCo_I2PTrainer(
        settings_path,
        fold=args.f,
        tversky_alpha=args.tversky_alpha,
        num_prototype=args.num_prototype,
        memory_rate=args.memory_rate,
        lambda_for_C=args.lambda_for_C,
        rampup_epoch=args.rampup_epoch,
    )
    Trainer.run_training()
