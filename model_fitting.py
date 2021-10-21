"""fitting
"""

import numpy as np
from numpy.polynomial import Polynomial
import linmix
import bambi
import pymc3 as pm
from scipy.optimize import minimize

from br_util.stats import robust_scatter

import broken_linear

# defaults to 5000, 2000 is good for our our data set, 500 for fast
LINMIX_MIN_ITR = 200


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


def rv_least_squares(data, fit_mask, high_degree_fits=False):
    linear_fit, lin_error = _least_squares(data, fit_mask, 1)
    if high_degree_fits:
        quadratic_fit, quad_error = _least_squares(data, fit_mask, 2)
        cubic_fit, cubic_error = _least_squares(data, fit_mask, 3)
    return linear_fit.convert()


def rv_broken_linear_bayesian(data):
    model = bambi.Model("x1_standardized ~ c", data)
    # pm.model_to_graphviz(model)
    return model.fit(draws=1000)


def rv_broken_linear_frequentist(data, fit_mask):
    # _chi_square(theta_cosmo, delta_theta, M_0, data, fitmask):
    res = minimize(
        broken_linear.chi_square,
        # theta_cosmo, delta_theta, M_0,
        x0=[np.arctan(3.1), 0, -19.4],
        args=(data, fit_mask),
        # x=data.data.loc[fit_mask, "c"],
        # y=data.data.loc[fit_mask, "x1_standardized"],
        # sigma=data.data.loc[fit_mask, "x1_standardized_ERR"],
        # absolute_sigma=False,  ## default value, but idk.
        # check_finite=True,
        bounds=(
            (np.arctan(1.0), np.arctan(5.0)),
            (np.arctan(-3.1), np.arctan(3.1)),
            (-20, -18),
        ),
        method=None,
    )
    return res


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
