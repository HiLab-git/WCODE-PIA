import torch
import itertools
import numpy as np
# from cleanlab.count import estimate_joint
import time


def correction_by_fn_estim(pred_logits: torch.Tensor, incomplete_label: torch.Tensor):
    """
    refine the incomplete label by estimating the FNs
    Inputs:
        pred_logits: predicted logits from teacher model in b, c=2, (z,) y, x
        incomplete_label: incomplete label in b, 1, (z,) y, x
    Outputs:
        refined_label: b, 1, (z,) y, x
    """
    b, c, *_ = pred_logits.shape
    assert c == 2, "Only binary prediction is supported."

    pred_prob = pred_logits.softmax(dim=1)

    with torch.no_grad():
        refined_label = []
        for idx_b in range(b):
            refined_label.append(
                # refine the label and add back the channel of batchsize
                correction(pred_prob[idx_b], incomplete_label[idx_b])[None]
            )
        refined_label = torch.cat(refined_label, dim=0)

    return refined_label


def correction(pred_soft: torch.Tensor, incomplete_label: torch.Tensor):
    """
    An implementation in batch-level
    Inputs:
        pred_soft: c, (z,) y, x
        incomplete_label: 1, (z,) y, x
    Outputs:
        return: 1, (z,) y, x
    """
    c, *spatial_size = pred_soft.shape

    # (z,) y, x
    incomplete_label = incomplete_label.squeeze().type(torch.float64)

    # startt = time.time()
    # using the values larger than the upper limitation of probability to initialize.
    confidence_threshold = (
        torch.zeros([c], device=pred_soft.device).type(torch.float64) + 2.0
    )

    for idx_c in range(c):
        cls_mask = incomplete_label == idx_c
        if cls_mask.any():
            confidence_threshold[idx_c] = (
                pred_soft[idx_c][cls_mask].mean().clip(min=2e-6)
            )

    # get the confidence joint matrix C
    confidence_joint = torch.zeros([c, c], device=pred_soft.device).float()
    for i, j in itertools.product(range(c), range(c)):
        confidence_joint[i][j] = (
            (incomplete_label == i) & (pred_soft[j] >= confidence_threshold[j])
        ).sum()

    # normalize the confidence joint matrix C
    labels_class_counts = confidence_joint.sum(dim=1, keepdims=True).float()  # 2, 1
    noise_matrix = confidence_joint / labels_class_counts.clip(min=1e-10)  # 2, 2
    noise_matrix = noise_matrix.clip(min=0.0, max=0.9999)
    # 2, 1
    X_card_i = torch.tensor(
        [(incomplete_label == idx_c).sum() for idx_c in range(c)],
        device=pred_soft.device,
    )[:, None].float()

    # confidence joint distribution Q
    Q_numer = noise_matrix * X_card_i
    Q_sum = Q_numer.sum()
    Q = Q_numer / Q_sum

    # print("Q cost:", time.time()-startt)
    # startt = time.time()

    # Q_cleanlab = estimate_joint(
    #     labels=incomplete_label.reshape(-1).cpu().numpy(),
    #     pred_probs=pred_soft.reshape(c, -1).transpose(0, 1).cpu().numpy(),
    # )
    # print("CleanLab cost:", time.time()-startt)

    # print(Q, Q_cleanlab)

    # estimated the number of FNs
    n_fn = int(Q[0][1] * np.prod(spatial_size))

    # find the threshold of probaility to define the FNs needed to be refined
    sorted_p0, _ = torch.sort(pred_soft[0, incomplete_label == 0].flatten())
    thres = sorted_p0[n_fn]

    # (z,) y, x
    pseudo_label = torch.zeros_like(incomplete_label, device=pred_soft.device)
    refined_region = (incomplete_label == 0) & (pred_soft[0] < thres)
    pseudo_label[refined_region] = 1
    pseudo_label = incomplete_label + pseudo_label

    # add channel of class back
    return pseudo_label[None]
