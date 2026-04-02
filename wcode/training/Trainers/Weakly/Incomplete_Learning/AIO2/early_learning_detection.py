import numpy as np
from scipy.optimize import curve_fit


def curve_func(x, a, b, c):
    return a * (1 - np.exp(-b * x**c))


def curve_gredient_func(x, a, b, c):
    return a * c * b * np.exp(-b * x**c) * (x ** (c - 1))


def linear_func(x, a, b):
    return a * x + b


def ACT_module(data: list, ngs_dict: dict, wsizes: list, detect_eps: np.ndarray):
    """
    Detect the resuming point I_r using warmup/transition ending point I_t and early-learning ending point I_e.
    Inputs:
        data: Accuracies [IoU, Dice or Accuracy] of training data during training.
        ngs_dict: Numerical gradients with different sliding window sizes.
        wsizes: List of sliding window sizes used to determine whether the training has passed the minimal gradients.
        detect_eps: Detected ending points of transition phase using different sliding window sizes.
    """
    if ngs_dict is None:
        ngs_dict = {b: [] for b in wsizes}

    if detect_eps is None:
        detect_eps = np.zeros(len(wsizes))

    check_buff = np.mean(wsizes)
    length_data = len(data)

    # calculate numerical gradients for points in the window size (step-2)
    for bi, ws in enumerate(wsizes):
        if length_data >= ws:
            x0 = np.arange(length_data - ws + 1, length_data + 1)
            y0 = np.array(data[length_data - ws : length_data])
            popt, _ = curve_fit(
                linear_func,
                x0,
                y0,
                p0=(1, 0),
                method="trf",
                bounds=([0, -np.inf], [np.inf, np.inf]),
            )
            a, b = tuple(popt)
            ngs_dict[ws].append(a)

            # check whether gradients start to decrease
            # detect the end point of the warmup/transition stage at window size ws
            if min(ngs_dict[ws]) < a:
                ind = np.argmin(ngs_dict[ws]) + ws
                if length_data - ind > check_buff:
                    detect_eps[bi] = ind

    # output final detection result
    try:
        if (detect_eps > 0).sum() == len(wsizes):
            # training has reached the end of transition stage
            I_t = int(
                np.mean(detect_eps)
            )  # final detected ending point of transition stage

            # fitting training accuracies (y) to exponential function
            x0 = np.arange(I_t) + 1
            y0 = np.array(data[:I_t])

            popt, _ = curve_fit(
                curve_func,
                x0,
                y0,
                p0=(1, 0.5, 0.5),
                method="trf",
                bounds=([0, 0, 0], [1, np.inf, 1]),
            )
            a, b, c = tuple(popt)

            # estimated y using fitted function
            yh = curve_func(x0, a, b, c)

            # adaptive threshold
            thr = (yh[-1] - yh[0]) / (x0[-1] - x0[0])

            # gradients of y (fitted function)
            yd = curve_gredient_func(x0, a, b, c)

            # transition phase starting point
            I_e = np.sum(yd > thr)

            # final detected trigger point
            I_r = int((I_t + I_e) / 2)
        else:
            I_r = 0
    except RuntimeError:
        I_r = 0

    return I_r, ngs_dict, detect_eps
