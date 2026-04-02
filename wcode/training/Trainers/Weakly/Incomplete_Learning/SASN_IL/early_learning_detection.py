import numpy as np
from scipy.optimize import curve_fit


def curve_func(x, a, b, c):
    return a * (1 - np.exp(-b * x**c))


def curve_gredient_func(x, a, b, c):
    return a * c * b * np.exp(-b * x**c) * (x ** (c - 1))


def find_t0(data: list, tau: float):
    """
    Detect the resuming point t0
    Inputs:
        data: Accuracies [IoU, Dice or Accuracy] of training data during training.
        tau: threshold in the condition.
    """
    length_data = len(data)
        
    # fit the exponential fuction
    x0 = np.arange(length_data)
    y0 = np.array(data)
    try:
        popt, _ = curve_fit(
            curve_func,
            x0,
            y0,
            p0=(1, 0.5, 0.5),
            method="trf",
            bounds=([0, 0, 0], [1, np.inf, np.inf]),
        )

        # caculate the gradient of this point
        gradient_2k = curve_gredient_func(2, *popt)

        # whether meet the condition
        for i in range(3, length_data):
            gradient_t = curve_gredient_func(i, *popt)
            # print(i, (np.abs(gradient_2k) - np.abs(gradient_t)) / (np.abs(gradient_2k)), tau)
            if np.abs(gradient_2k) >= 1e-5:
                if (np.abs(gradient_2k) - np.abs(gradient_t)) / (np.abs(gradient_2k)) > tau:
                    t0 = i
                    break
    except RuntimeError:
        t0 = length_data
    except UnboundLocalError:
        t0 = length_data
        
    return t0
