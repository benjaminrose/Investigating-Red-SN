"""Functions for hypothesis testing.

Is A consistant with B?
"""
import numpy as np
from scipy.stats import ks_2samp

from KS2D.KS2D import ks2d2s


def ks2d(x1, y1, x2, y2):
    N_samples = 150
    index1 = np.random.randint(0, len(x1), N_samples)
    index2 = np.random.randint(0, len(x2), N_samples)
    # if full:
    #     step1 = 1
    #     step2 = 1
    # else:
    #     step1 = 20
    #     step2 = 4950 if fig_options.get("sim_name") == "BS21" else 370
    _, ks_p = ks2d2s(
        np.stack((x1[index1], y1[index1]), axis=1),
        np.stack((x2[index2], y2[index2]), axis=1),
    )
    return ks_p, len(x2[index2])  # just make sure N_samples is the actual length


def ks1d(a, b):
    return ks_2samp(a, b)[1]
