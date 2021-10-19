"""Functions for hypothesis testing.

Is A consistant with B?
"""
import numpy as np
from scipy.stats import ks_2samp

from KS2D.KS2D import ks2d2s


def ks2d(x1, y1, x2, y2, fig_options, full=False):
    print("Running KS2D ...")
    if full:
        step1 = 1
        step2 = 1
    else:
        step1 = 20
        step2 = 4950 if fig_options.get("sim_name") == "BS21" else 370
    _, ks_p = ks2d2s(
        np.stack((x1[::step1], y1[::step1]), axis=1),
        np.stack((x2[::step2], y2[::step2]), axis=1),
    )
    return ks_p, step1, step2


def ks1d(a, b):
    return ks_2samp(a, b)[1]
