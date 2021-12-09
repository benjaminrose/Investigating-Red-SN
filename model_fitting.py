"""fitting
"""

from colorama import Fore, Style
import numpy as np
from numpy.polynomial import Polynomial
import linmix
import bambi
import pymc3 as pm
import arviz as az
from theano import tensor as tt
from scipy.optimize import minimize

from br_util.stats import robust_scatter

import broken_linear

# defaults to 5000, 2000 is good for our our data set, 500 for fast
LINMIX_MIN_ITR = 1000
PYMC_TUNE = 1000
PYMC_DRAWS = 750
PYMC_CHAINS = 4


def rv_full_range(data, fit_mask):
    lm = linmix.LinMix(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        xsig=data.data.loc[fit_mask, "cERR"],
        ysig=data.data.loc[fit_mask, "x1_standardized_ERR"],
    )
    lm.run_mcmc(miniter=LINMIX_MIN_ITR)
    print(
        Fore.BLUE
        + f"* with LINMIX: Beta = {np.median(lm.chain['beta']):.3f} +/- {robust_scatter(lm.chain['beta']):.3f}"
        + Style.RESET_ALL
    )
    return lm.chain


def rv_least_squares(data, fit_mask, high_degree_fits=False):
    linear_fit, lin_error = _least_squares(data, fit_mask, 1)
    if high_degree_fits:
        quadratic_fit, quad_error = _least_squares(data, fit_mask, 2)
        cubic_fit, cubic_error = _least_squares(data, fit_mask, 3)
    return linear_fit.convert()


def rv_broken_linear_bayesian(data):
    """PyMC3's normal and skew normal distributions can be passed the keyword
    `sigma` to use the standard deviation scale parameter. This is what we did
    in the code, however our notation in the paper ($\mathcal{N} \sim (\mu, \sigma^2)$)
    uses the more common notational practice of using the variance scale parameter.
    """
    # model = bambi.Model("x1_standardized ~ c", data)
    # model.build()
    # model.graph(name="model", fmt="pdf")
    # return model.fit(draws=1000)

    coords = {"observation": np.arange(data.shape[0]), "vars": np.arange(2)}
    with pm.Model(coords=coords) as model:
        # Model Variables
        theta_cosmo = pm.Uniform(
            "theta_cosmo", lower=1.2, upper=1.45
        )  # mu=1.258, sd=0.1)
        delta_theta = pm.Normal("delta_theta", mu=0, sigma=0.15)
        M0 = pm.Uniform("M0", lower=-19.4, upper=-19.2)
        sigma_int = pm.HalfCauchy(
            "sigma_int", 0.05
        )  # variance of epsilon distribution in Kelly2007 eqn 1.

        # Hyper parameters
        # sigma_squared = pm.Uniform("sigma^2", 0, 0.1)
        c_pop_mean = pm.Cauchy("c_pop_mean", alpha=0, beta=0.3)  # 0
        c_pop_scatter = pm.Uniform("c_pop_sigma", lower=0.01, upper=0.2)  # 0.1
        c_pop_skewness = pm.Uniform("c_pop_skewness", lower=-0.1, upper=2.0)  # 0.1

        # "True space" parameters
        c_true = pm.SkewNormal(  # I don't need LINMIX's Gaussian Mixture, I know I have a SkewNormal and PPC verifies this assumption
            "c_true",
            mu=c_pop_mean,
            sigma=c_pop_scatter,
            alpha=c_pop_skewness,
            dims="observation",
        )
        c_obs = pm.Normal(
            "c",
            mu=c_true,
            sigma=data["cERR"],
            observed=data["c"],
        )
        M_true = pm.Normal(
            "M'_true",
            # broken_linear needs an observed variable to cut on
            mu=broken_linear.broken_linear(c_true, theta_cosmo, delta_theta, M0),
            sigma=sigma_int,
            dims="observation",
        )

        # Observed space parameters
        # shape == (observations, 2) or (obs, 2, 2) for cov
        ## cov = np.zeros((len(data["x1_standardized_ERR"]), 2, 2))
        ## cov[:, 0, 0] = data["x1_standardized_ERR"]
        ## cov[:, 1, 1] = data["cERR"]
        # M_err = pm.Data("M'_err", data["x1_standardized_ERR"], dims="observation")
        # c_err = pm.Data("c_err", data["cERR"], dims="observation")
        # # cov = tt.stack(([M_err, 0], [0, c_err]))
        # # cov = tt.stack((M_err, c_err))
        # # breakpoint()
        # cov = np.array(
        #     [[1, 0], [0, 1]]
        # )  # cov = tt.stack([[M_err, np.zeros(1808)], [np.zeros(1808), c_err]])
        # # chol, _, _ = pm.LKJCholeskyCov('chol_cov', n=1808, eta=1, sd_dist=sd, compute_corr=True)
        # # no off diagonal terms
        # mu = tt.stack([M_true, c_true], axis=1)
        # obs = np.stack((data["x1_standardized"], data["c"]), axis=1)
        # # vals = pm.MvNormal("vals", mu=mu, cov=cov, shape=(5, 2))
        # vals = pm.MvNormal(
        #     "vals",
        #     mu=mu,
        #     cov=cov,
        #     observed=obs,  # , dims=("observation", "vars"), shape=(1808,2)
        # )
        M_obs = pm.Normal(
            "M'",
            mu=M_true,
            sd=data["x1_standardized_ERR"],
            observed=data["x1_standardized"],
        )

        # Save model
        # adding formatting="plain_with_params" does not work well with pm.Deterministic
        graph = pm.model_to_graphviz(model)
        graph.render(filename="figures/model", format="pdf")
        # Sample model
        trace = pm.sample(PYMC_DRAWS, chains=PYMC_CHAINS, tune=PYMC_TUNE)
        prior = pm.sample_prior_predictive()
        posterior_predictive = pm.sample_posterior_predictive(trace)
        pm_data = az.from_pymc3(
            trace=trace,
            prior=prior,
            posterior_predictive=posterior_predictive,
            # coords={"school": np.arange(eight_school_data["J"])},
            # dims={"theta": ["school"], "theta_tilde": ["school"]},
        )
    return pm_data


def rv_broken_linear_frequentist(data, fit_mask):
    res = minimize(
        # chi_square(params, data, fit_mask):
        # params = theta_cosmo, delta_theta, M_0
        broken_linear.chi_square,
        x0=[np.arctan(3.1), 0, -19.4],
        args=(data, fit_mask),
        bounds=(
            (np.arctan(0.5), np.arctan(6.0)),
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
