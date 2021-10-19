"""fitting
"""

import numpy as np
from numpy.polynomial import Polynomial
import linmix

from br_util.stats import robust_scatter

# defaults to 5000, 2000 is good for our our data set, 500 for fast
LINMIX_MIN_ITR = 200


def _broken_linear(c, m_cosmo, m_red, M_0, c_break=0.3):
    """define a broken linear color-luminosity relationship

    Parameters
    ----------
    c : np.array
        Color (x-)values
    m_cosmo : float
        slope for c <= c_break.
    m_red : float
        slope for c_break < c.
    M_0 : float
        The intercept (c=0) value
    c_break : float
        Break point in the broken linear function. Defaults to 0.3.

    Return
    ------
    mag : np.array
        Standardized (before color corretions) absolute magnitude (y-)values.
    """
    return np.piecewise(
        c,
        [c <= c_break, c > c_break],
        [lambda x: m_cosmo * x + M_0, lambda x: m_red * x + M_0],
    )


def rv_full_range(data, fit_mask):
    lm = linmix.LinMix(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        xsig=data.data.loc[fit_mask, "cERR"],
        ysig=data.data.loc[fit_mask, "x1_standardized_ERR"],
    )
    lm.run_mcmc(miniter=LINMIX_MIN_ITR)
    print(
        f"\n* with LINMIX: Beta = {np.median(lm.chain['beta']):.3f} +/- {robust_scatter(lm.chain['beta']):.3f}"
    )
    return lm.chain


def rv_broken_linear():
    pass


def rv_least_squares(data, fit_mask, high_degree_fits=False):
    linear_fit, lin_error = _least_squares(data, fit_mask, 1)
    if high_degree_fits:
        quadratic_fit, quad_error = _least_squares(data, fit_mask, 2)
        cubic_fit, cubic_error = _least_squares(data, fit_mask, 3)
    return linear_fit.convert()


def _least_squares(data, fit_mask, deg):
    # if cERR << x1_standardized_ERR
    # use .convert cause numpy's default is dumb.
    # https://numpy.org/doc/stable/reference/routines.polynomials.html#transition-guide
    fit, error = Polynomial.fit(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        w=data.data.loc[fit_mask, "x1_standardized_ERR"],
        full=True,
        deg=deg,
    )
    return fit, error
