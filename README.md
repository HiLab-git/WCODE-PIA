# WCODE-PIA: Partial-instance Annotation

**WCODE-PIA**, that focuses on the learning of incomplete annotations, is a medical image segmentation framework improved from [**WCODE**](https://github.com/WltyBY/WCODE).

* This project focuses on the incomplete labeling task, in which the foreground area is partially labeled and the remaining pixels are considered as the background.
  

---
## 📖 Our works
|Title|Implementation|Web|
|---|---|---|
|Weakly Supervised Lymph Nodes Segmentation Based on Partial Instance Annotations with Pre-trained Dual-branch Network and Pseudo Label Learning|[DBDMP](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers/Weakly/Incomplete_Learning/DBDMP)|[MELBA2024](https://www.melba-journal.org/papers/2024:017.html)|
|ReCo-I2P: An Incomplete Supervised Lymph Node Segmentation Framework Based on Orthogonal Partial-Instance Annotation|[ReCo-I2P](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P)|MICCAI2025 (Oral)|


---
## 🔬 Related Literatures
Some implementations of compared state-of-the-art (SOTA) methods can be found [**here**](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers).

**IA** - Inaccurate label, **IC** - Incomplete label
|Category|Authors|Title|Implementation|Web|
|---|---|---|---|---|
|**IA**|B. Han et al.|Co-teaching: robust training of deep neural networks with extremely noisy labels|[Coteaching](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers/Weakly/Incomplete_Learning/ReCo_I2P)|[NeurIPS2018](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)|
|**IA**|C. Fang et al.|Reliable Mutual Distillation for Medical Image Segmentation Under Imperfect Annotations|[RMD](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers/Weakly/NLL/RMD)|[TMI2023](https://ieeexplore.ieee.org/abstract/document/10021263)|
|**IA**|T. Weng et al.|Accurate Segmentation of Optic Disc and Cup from Multiple Pseudo-labels by Noise-aware Learning|[MPNN](https://github.com/HiLab-git/WCODE-PIA/tree/main/wcode/training/Trainers/Weakly/NLL/MPNN)|[CSCWD2024](https://ieeexplore.ieee.org/abstract/document/10580087)|
|**IC**|C. Liu et al.|AIO2: Online Correction of Object Labels for Deep Learning With Incomplete Annotation in Remote Sensing Image Segmentation|None|[TGRS2024](https://ieeexplore.ieee.org/abstract/document/10460569)

<!-- --- -->
<!-- ## ⛺ Discussion of the task setting and general usage -->


---
## ✉️ Contact
--- Email: litingyuwang@gmail.com