"""module contianing all the functions around the borken linear model."""

import numpy as np


def broken_linear(c, theta_cosmo, delta_theta, M_0, c_break=0.3):
    """define a broken linear color-luminosity relationship

    Parameters
    ----------
    c : pandas.Series
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
    m_cosmo = np.tan(theta_cosmo)
    delta_m = np.tan(delta_theta)
    return np.piecewise(
        c.values,
        c.le(c_break),
        [lambda x: m_cosmo * x + M_0, lambda x: (m_cosmo + delta_m) * x + M_0],
    )


def chi_square_params2slopes(params):
    """Take a scipy.optimize.minimize `param` tuple and convert to common variables.

    This is for `_chi_square()`. `params` is `(theta_cosmo, delta_theta, M_0)` from
    `_borken_lienear`. `theta_cosmo` and `delta_theta` are converted to `beta` and
    `delta_beta` via the `np.tan` function.
    """
    theta_cosmo, delta_theta, M_0 = params
    return np.tan(theta_cosmo), np.tan(delta_theta), M_0


def chi_square(params, data, fit_mask):
    """
    theta: list
        Iterable containing the arguments to `_broken_linear`.
    """
    theta_cosmo, delta_theta, M_0 = params
    numerator = (
        broken_linear(data.data.loc[fit_mask, "c"], theta_cosmo, delta_theta, M_0)
        - data.data.loc[fit_mask, "x1_standardized"]
    ) ** 2
    squared_errors = (
        data.data.loc[fit_mask, "cERR"] ** 2
        + data.data.loc[fit_mask, "x1_standardized_ERR"] ** 2
    )
    return np.sum(numerator / squared_errors)


def log_prior(theta):
    alpha, beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0) else:
    return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)


def log_like(theta, x, y):
    alpha, beta, sigma = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(
        np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2
    )


def log_posterior(theta, x, y):
    return log_prior(theta) + log_like(theta, x, y)
