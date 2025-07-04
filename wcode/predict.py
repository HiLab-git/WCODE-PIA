import os
import argparse

from wcode.inferring.PatchBasedPredictor import PatchBasedPredictor
from wcode.inferring.Evaluator import Evaluator
from wcode.utils.file_operations import open_yaml

GPU_ID = 0

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="LNQ2023", help="Name of dataset")
parser.add_argument(
    "--settiing_file",
    type=str,
    default="LNQ2023Sparse_Fully_0.yaml",
    help="Name of setting files, or its absolute path",
)

parser.add_argument(
    "-i",
    type=str,
    default="./Dataset/LNQ2023/imagesTs",
    help="folder path of input images",
)
parser.add_argument(
    "--gt_path",
    type=str,
    default="./Dataset/LNQ2023/labelsTs",
    help="Path of ground truth. If given, will do evaluation after prediction",
)

parser.add_argument(
    "-o",
    type=str,
    default="./Logs/LNQ2023/Ours_v2.2_MICCAI_loss/tversky_alpha_0.3_consis_weight_0.3_rampup_epoch_100_update_way_all_select_way_most_num_prototype_3_memory_rate_0.999/fold_0/test_final",
    help="folder path of save path",
)
parser.add_argument(
    "-m",
    type=str,
    default="./Logs/LNQ2023/Ours_v2.2_MICCAI_loss/tversky_alpha_0.3_consis_weight_0.3_rampup_epoch_100_update_way_all_select_way_most_num_prototype_3_memory_rate_0.999/fold_0/checkpoint_final.pth",
    help="saving path of using model",
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument("-s", type=str, default=None, help="split of data, can be None")
parser.add_argument(
    "--data_dim",
    type=str,
    default="3d",
    help="dim of data, 2d or 3d",
)
parser.add_argument(
    "--save_probabilities",
    type=bool,
    default=False,
    help="Save the predicted probabilities or not.",
)
args = parser.parse_args()


if __name__ == "__main__":
    config_dict = open_yaml(os.path.join("./Configs", args.settiing_file))

    predict_configs = {
        "dataset_name": args.dataset,
        "modality": "all",
        "fold": args.f,
        "split": args.s,
        "original_img_folder": args.i,
        "predictions_save_folder": args.o,
        "model_path": args.m,
        "device": {"gpu": [GPU_ID]},
        "overwrite": True,
        "save_probabilities": args.save_probabilities,
        "patch_size": config_dict["Training_settings"]["patch_size"],
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "perform_everything_on_gpu": True,
        "use_mirroring": True,
        "allowed_mirroring_axes": [0, 1] if args.data_dim == "2d" else [0, 1, 2],
        "num_processes": 16,
    }

    config_dict["Inferring_settings"] = predict_configs
    predictor = PatchBasedPredictor(config_dict, allow_tqdm=True)
    predictor.initialize()
    predictor.predict_from_file(
        predictor.original_img_folder,
        predictor.predictions_save_folder,
        predictor.modality,
        args.data_dim,
    )
    if isinstance(args.gt_path, str):
        evaluator = Evaluator(
            prediction_folder=args.o,
            ground_truth_folder=args.gt_path,
            dataset_yaml_or_its_path=f"./Dataset_preprocessed/{args.dataset}/dataset.yaml",
        )
        evaluator.run()
