import numpy as np
import torch
import random
from .torch_deform import deform_grid


def gaussian_noise(img, mean=0, std=0.001):
    noise = torch.normal(mean, std, size=img.size(), device=img.device)
    return img + noise


def flip(img, flipCode=None):
    """
    img: input ,tensor
    flipCode: flip or not
    """
    dim = len(img.shape)
    if flipCode is None:
        axes = [i for i in range(2, dim)]
        num = random.randint(1, len(axes))
        flipCode = random.sample(axes, k=num)

    img = torch.flip(img, flipCode)
    return img, flipCode


def deform(img, displacements=None, rotates=None, zooms=None):
    n = img.shape[0]
    if displacements is None:
        displacements = []
    if rotates is None:
        rotates = []
    if zooms is None:
        zooms = []
    imgnew = []
    for i in range(n):
        imgtmp = img[i]
        if len(displacements) < n:
            num = random.random() * 25 + 1
            displacement = np.random.randn(2, 3, 3) * num
            displacements.append(displacement)
        else:
            displacement = displacements[i]
        if len(rotates) < n:
            rotate = np.random.uniform(0, 60)
            rotates.append(rotate)
        else:
            rotate = rotates[i]
        if len(zooms) < n:
            zoom = np.random.uniform(1, 2)
            zooms.append(zoom)
        else:
            zoom = zooms[i]
        imgnewtmp = deform_grid(
            imgtmp,
            torch.Tensor(displacement),
            order=3,
            mode="nearest",
            rotate=rotate,
            zoom=zoom,
            axis=(1, 2),
        )
        imgnew.append(imgnewtmp)
    return torch.stack(imgnew, 0), displacements, rotates, zooms
