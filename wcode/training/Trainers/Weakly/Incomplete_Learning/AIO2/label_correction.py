import torch
import numpy as np
import torch.nn.functional as F
from scipy import ndimage


def generate_soft_edge(mask: torch.Tensor, filter_size: int, device: str):
    """
    Inputs:
        mask: label needed softened, in b, c, (z,) y, x
        filter_size: size of convoluntional filter
    """
    dim = mask.ndim
    assert dim == 4 or dim == 5, "Wrong size of mask in soft label generation!"
    mask = mask.to(device)
    # filter
    filters = (
        torch.ones(1, 1, *[filter_size for _ in range(dim - 2)]).float().to(device)
    )
    if dim - 2 == 2:
        new_mask = F.conv2d(mask, filters, padding="same")
    elif dim - 2 == 3:
        new_mask = F.conv3d(mask, filters, padding="same")
    else:
        raise ValueError("Unsupport dimension of data.")
    new_mask = new_mask / (filter_size**2)

    return new_mask * mask


def object_wise_label_correction(
    noisy_label: np.ndarray,
    hard_pred: np.ndarray,
    device: str,
    filter_size: int = -1,
    filter_all: bool = False,
    dataset_type: str = "Dense",
):
    """
    Only work for two-class (background and one foreground) segmentation task
    Inputs:
        noisy_label: incomplete/noisy label in b, 1, (z,) y, x
        hard_pred: hard prediction of model in b, 1, (z,) y, x
        filter_size: size of convolutional filter to soften the label
        filter_all: filter all the instance (True) or newly detected ones (False).
    """
    dim = hard_pred.ndim - 2
    assert hard_pred.shape[1] == 1
    assert dataset_type in ["Dense", "Sparse"]

    instance_new_mask = []
    for b in range(hard_pred.shape[0]):
        # (z,) y, x
        pred_b = hard_pred[b].squeeze().astype(int)
        noisy_b = noisy_label[b].squeeze().astype(float)
        noisy_binary_mask = np.zeros_like(noisy_b)
        noisy_binary_mask[noisy_b != 0] = 1

        # label out connected components
        s = ndimage.generate_binary_structure(dim, connectivity=1)
        labeled_array, numpatches = ndimage.label(pred_b, s)

        # check overlap, remain the objects not overlapped with noisy_label
        overlap_inds = np.unique(labeled_array * noisy_binary_mask)
        for ind in overlap_inds:
            labeled_array[labeled_array == ind] = 0

        labeled_array[labeled_array != 0] = 1

        # add the channel c=1 back
        instance_new_mask.append(labeled_array[None])
    if dataset_type == "Dense":
        # stack the channel b back and convert to Tensor
        instance_new_mask = torch.from_numpy(np.stack(instance_new_mask, axis=0)).float()
    else:
        instance_new_mask = torch.from_numpy(hard_pred).float()
    noisy_label = torch.from_numpy(noisy_label).float()

    if filter_size > 0:
        # soften the edge
        if filter_all:
            # convert all the instances (instances in both noisy_label and instance_new_mask)
            ## update noisy labels
            new_mask = noisy_label + instance_new_mask
            ## clip the values greater than 1. In theory, there shouldn't be a value greater than 1 here, but who knows.
            y = torch.ones_like(new_mask)
            new_mask = torch.where(new_mask <= 1, new_mask, y)
            ## generate soft labels for all instances
            new_mask = generate_soft_edge(new_mask, filter_size, device)
        else:
            # only convert the newly predicted instances
            ## generate soft labels for new ones
            instance_new_mask = generate_soft_edge(
                instance_new_mask, filter_size, device
            )
            # update noisy labels
            new_mask = noisy_label.to(device) + instance_new_mask
            ## clip the values greater than 1.
            y = torch.ones_like(new_mask, device=device)
            new_mask = torch.where(new_mask <= 1, new_mask, y)
    else:
        # use hard labels
        new_mask = noisy_label + instance_new_mask
        y = torch.ones_like(new_mask)
        new_mask = torch.where(new_mask <= 1, new_mask, y)
        new_mask = new_mask.to(device)

    return new_mask


def pixel_wise_label_correction(noisy_label, pred_logits, confidence=0.8):
    """
    Only work for two-class (background and one foreground) segmentation task
    """
    pred_probs = torch.softmax(pred_logits, dim=1)
    fg_confidence = pred_probs[:, 1:]
    new_mask = torch.zeros_like(noisy_label)
    new_mask[fg_confidence > confidence] = 1
    new_mask[noisy_label != 0] = 1
    return new_mask


############### old ones ###############
# def pixel_wise_label_correction(noisy_label, pred_logits, confidence=0.8):
#     pred_probs = torch.softmax(pred_logits, dim=1)
#     noisy_label[pred_probs > confidence] = 1
#     noisy_label[pred_probs < 1 - confidence] = 0
#     return noisy_label
