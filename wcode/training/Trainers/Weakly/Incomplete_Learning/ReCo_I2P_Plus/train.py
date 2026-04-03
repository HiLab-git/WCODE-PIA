import os
import argparse

from wcode.training.Trainers.Weakly.Incomplete_Learning.ReCo_I2P_Plus.TestTrainer import (
    TestTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting",
    type=str,
    default=None,
    help="File Name of Setting yaml, or you can just give the absolute path of the file.",
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--tversky_alpha", type=float, default=0.3, help="alpha of tversky loss"
)
parser.add_argument(
    "--AwCE_beta", type=float, default=1.0, help="beta of AwCE"
)
parser.add_argument(
    "--consis_weight", type=float, default=1.0, help="consistency weight of two preds"
)
parser.add_argument(
    "--rampup_epoch", type=int, default=100, help="rampup epoch of consis_weight"
)
parser.add_argument(
    "--update_way",
    type=str,
    default="least",
    help="the way to update memoried prototypes, least or all.",
)
parser.add_argument(
    "--select_way",
    type=str,
    default="merge",
    help="the way to select memoried prototypes, most or merge.",
)
parser.add_argument(
    "--num_prototype", type=int, default=3, help="memoried inter-batch prorotypes."
)
parser.add_argument(
    "--memory_rate",
    type=float,
    default=0.999,
    help="memoried rate of inter-batch prorotypes.",
)
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = TestTrainer(
        settings_path,
        args.f,
        tversky_alpha=args.tversky_alpha,
        awce_beta=args.AwCE_beta,
        consis_weight=args.consis_weight,
        rampup_epoch=args.rampup_epoch,
        update_way=args.update_way,
        select_way=args.select_way,
        num_prototype=args.num_prototype,
        memory_rate=args.memory_rate,
    )
    Trainer.run_training()
