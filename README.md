# WCODE-PIA: Partial-instance Annotation

**WCODE-PIA**, which focuses on the learning of incomplete annotations, is a medical image segmentation framework improved from [**WCODE**](https://github.com/WltyBY/WCODE)-v0.

* This project focuses on the incomplete labeling task, in which the foreground area is partially labeled, and the remaining pixels are considered as the background.
  

---
## 📖 Our works
|Title|Implementation|Web|
|---|:-:|:-:|
|Weakly Supervised Lymph Nodes Segmentation Based on Partial Instance Annotations with Pre-trained Dual-branch Network and Pseudo Label Learning|[DBDMP](/wcode/training/Trainers/Weakly/Incomplete_Learning/DBDMP)|[MELBA2024](https://www.melba-journal.org/papers/2024:017.html)|
|ReCo-I2P: An Incomplete Supervised Lymph Node Segmentation Framework Based on Orthogonal Partial-Instance Annotation|[ReCo-I2P](/wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P)|[MICCAI2025 (Oral)](https://link.springer.com/chapter/10.1007/978-3-032-05169-1_49)|
|Learning from 3D Partial Foreground Annotations: Prototype-Enhanced Incomplete Supervision for Lymph Node Segmentation|[ReCo-I2P+](/wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P_Plus)|-|


---
## 🔬 Related Literatures
Some implementations of compared state-of-the-art (SOTA) methods can be found [**here**](/wcode/training/Trainers).

**IA** - Inaccurate label, **IC** - Incomplete label

|Category|Authors|Title|Implementation|Web|
|:-:|---|---|:-:|:-:|
|**IA**|B. Han et al.|Co-teaching: robust training of deep neural networks with extremely noisy labels|[Coteaching](/wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P)|[NeurIPS2018](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)|
|**IA**|C. Fang et al.|Reliable Mutual Distillation for Medical Image Segmentation Under Imperfect Annotations|[RMD](/wcode/training/Trainers/Weakly/NLL/RMD)|[TMI2023](https://ieeexplore.ieee.org/abstract/document/10021263)|
|**IA**|T. Weng et al.|Accurate Segmentation of Optic Disc and Cup from Multiple Pseudo-labels by Noise-aware Learning|[MPNN](/wcode/training/Trainers/Weakly/NLL/MPNN)|[CSCWD2024](https://ieeexplore.ieee.org/abstract/document/10580087)|
|**IC**|C. Liu et al.|AIO2: Online Correction of Object Labels for Deep Learning With Incomplete Annotation in Remote Sensing Image Segmentation|[AIO2](/wcode/training/Trainers/Weakly/Incomplete_Learning/AIO2)|[TGRS2024](https://ieeexplore.ieee.org/abstract/document/10460569)|
|**IC**|H. Zhou et al.|Unsupervised domain adaptation for histopathology image segmentation with incomplete labels|[SASN_IL](/wcode/training/Trainers/Weakly/Incomplete_Learning/SASN_IL)|[CBM2024](https://www.sciencedirect.com/science/article/abs/pii/S001048252400310X)|

<!-- --- -->
<!-- ## ⛺ Discussion of the task setting and general usage -->


---
## 💾 Dataset and Related Weight of Models

We only provide the preprocessed dataset used in the experiment; see [this file](/Dataset/README.md) for details.

### Related weight of models

| Dataset         |   DSC (%)   |  ASSD (mm)  |  I-F1 (%)   |
| --------------- | :---------: | :---------: | :---------: |
| LNQ2023 ($P_1$) | 57.97±15.90 | 10.88±11.40 | 31.66±14.83 |

LNQ2023 - BaiduNetdisk: https://pan.baidu.com/s/1vyDE5N51vtCqLFIXH-Srcw?pwd=0319

## 🚀 Quick Start / Usage

### Preparation of Python Environment

Create a Python environment using conda.

```bash
# Create virtual environment
conda create -n wcode python=3.11

conda activate wcode
```

You can install PyTorch first from [the official website]([PyTorch](https://pytorch.org/)). In our implementation, we adapt PyTorch 2.5.1. Then, install additional dependencies.

```bash
pip install -r ./wcode/requirements.txt
```

### Data Preprocessing

1. Use the scripts in `./wcode/convert_datasets` to convert the dataset into a format compatible with the [WCODE](https://github.com/WltyBY/WCODE) repository, detailed in [`support_dataset_format.md`](https://github.com/WltyBY/WCODE/blob/main/documentation/support_dataset_format.md). If you use the dataset we provide, this step is unnecessary.
2. Run `./wcode/data_analysis_and_preprocess.py` in this repository to preprocess the dataset (details are in [`dataset_analysis_and_preprocess.md`]([dataset_analysis_and_preprocess.md](https://github.com/WltyBY/WCODE/blob/main/documentation/dataset_analysis_and_preprocess.md))).

```bash
# a quick start
PYTHONPATH=. python3 wcode/data_analysis_and_preprocess.py --dataset LNQ2023 --preprocess_config 3d

# using PYTHONPATH=. python3 wcode/data_analysis_and_preprocess.py -h to see all the params.
```

The preprocessed data will be saved in `./Dataset_preprocessed`.

### Training

The training scripts are saved in the `train.py` file within each method's implementation directory in [`./wcode/training/Trainers`](/wcode/training/Trainers).

```bash
# Run ReCo-I2P
PYTHONPATH=. python3 wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P/train.py --name_setting LNQ2023_ReCo_I2P.yaml -f 0 
```

### Predicting

Under the `./wcode` directory, three prediction scripts start with "predict" are used for inference: `predict.py`, `predict_for_unregistered_model.py`, and `predict_data_from_another_dataset.py`. Among them, the first two scripts are required for normal use cases: the first is for direct inference with models provided by **WCODE** in [`./wcode/net/build_network.py`](/wcode/net/build_network.py), while the second is for custom models.

## 📚 Citation

```
@article{wang2024weakly,
  title={Weakly Supervised Lymph Nodes Segmentation Based on Partial Instance Annotations with Pre-trained Dual-branch Network and Pseudo Label Learning},
  author={Wang, Litingyu and Qu, Yijie and Luo, Xiangde and Liao, Wenjun and Zhang, Shichuan and Wang, Guotai},
  journal={Machine Learning for Biomedical Imaging},
  volume={2},
  note={MICCAI 2023 LNQ challenge special issue},
  pages={1030--1047},
  year={2024}
}

@inproceedings{wang2025reco,
  title={ReCo-I2P: An Incomplete Supervised Lymph Node Segmentation Framework Based on Orthogonal Partial-Instance Annotation},
  author={Wang, Litingyu and Ye, Ping and Liao, Wenjun and Zhang, Shichuan and Zhang, Shaoting and Wang, Guotai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={507--517},
  year={2025},
  organization={Springer}
}
```

## ✉️ Contact

--- Email: litingyuwang@gmail.com