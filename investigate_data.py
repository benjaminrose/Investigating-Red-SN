""" investigate_red_sn.py

Requires python 3.9
"""

from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

import linmix

from br_util.stats import robust_scatter


from fitres import Fitres
from figures import *

from util import parse_cli, __version__


def broken_beta(c, m_cosmo, m_red, M_0, c_break=0.3):
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


if __name__ == "__main__":
    cli = parse_cli()

    # GLOBAL DEFAULTS
    #################
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose

    RUN_LINMIX = cli.linmix
    # defaults to 5000, 2000 is good for our our data set, 500 for fast
    LINMIX_MIN_ITR = 200
    C_MAX_FIT = 2.0  # cli.cmax # this is currently defined in several locations.

    c_splits = [0.3]  # was [0.1, 0.3] during initial analysis

    x1err_max = 1.5
    x1_max = 3  # x1 cut is on abs(x1)
    cerr_max = 0.15
    c_min = -0.3
    # fitprob_min = 0.1

    alpha = cli.alpha

    # Import and clean data
    #########################
    print("")  # provides spacing in outputs
    print("# Investigating Highly Reddened SN\n")

    data = Fitres(data_file, alpha, VERBOSE)
    data.clean_data(x1err_max, x1_max, cerr_max, c_min)
    data.calc_HR()

    print("## Demographics of Sample\n")
    print("### Number of Highly Reddened SN\n")
    print(f"There are {data.data.loc[data.data['c']>0.3, 'c'].shape[0]} SN with c>0.3")
    print(
        f"There are {data.data.loc[np.logical_and(data.data['c']>0.3, data.data['c']<=1.0), 'c'].shape[0]} SN with 0.3 < c <= 1.0"
    )
    print(
        f"There are {data.data.loc[data.data['c']>1.0, 'c'].shape[0]} SN with c > 1.0 \n"
    )

    data.slipt_on_c(0.99)
    print("### SN affected by c=1 boundry.")
    print(
        data.red_subsample[
            [
                "CIDint",
                "IDSURVEY",
                "c",
                "cERR",
                "x1_standardized",
                "x1_standardized_ERR",
            ]
        ]
    )

    print("\n## Basic Fit of Color-Luminosity Relationship")
    fit_mask = data.data["c"] <= C_MAX_FIT
    if RUN_LINMIX:
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
        fit = lm.chain

        # fit_mask_cosmo = (-0.3 <= data.data["c"]) & (data.data["c"] <= 0.3)
        # lm_cosmo = linmix.LinMix(
        #     x=data.data.loc[fit_mask_cosmo, "c"],
        #     y=data.data.loc[fit_mask_cosmo, "x1_standardized"],
        #     xsig=data.data.loc[fit_mask_cosmo, "cERR"],
        #     ysig=data.data.loc[fit_mask_cosmo, "x1_standardized_ERR"],
        # )
        # lm_cosmo.run_mcmc(miniter=LINMIX_MIN_ITR)
        # print(
        #     f"\n* with LINMIX: Beta_cosmo = {np.median(lm_cosmo.chain['beta']):.3f} +/- {robust_scatter(lm_cosmo.chain['beta']):.3f}"
        # )

        # fit_mask_red = 0.3 < data.data["c"]
        # lm_red = linmix.LinMix(
        #     x=data.data.loc[fit_mask_red, "c"],
        #     y=data.data.loc[fit_mask_red, "x1_standardized"],
        #     xsig=data.data.loc[fit_mask_red, "cERR"],
        #     ysig=data.data.loc[fit_mask_red, "x1_standardized_ERR"],
        # )
        # lm_red.run_mcmc(miniter=LINMIX_MIN_ITR)
        # print(
        #     f"\n* with LINMIX: Beta_red = {np.median(lm_red.chain['beta']):.3f} +/- {robust_scatter(lm_red.chain['beta']):.3f}"
        # )
    else:
        lm = None
        fit = None

    # if cERR << x1_standardized_ERR
    # use .convert cause numpy's default is dumb.
    # https://numpy.org/doc/stable/reference/routines.polynomials.html#transition-guide
    linear_fit, lin_error = Polynomial.fit(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        w=data.data.loc[fit_mask, "x1_standardized_ERR"],
        full=True,
        deg=1,
    )
    quadratic_fit, quad_error = Polynomial.fit(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        w=data.data.loc[fit_mask, "x1_standardized_ERR"],
        full=True,
        deg=2,
    )
    cubic_fit, cubic_error = Polynomial.fit(
        x=data.data.loc[fit_mask, "c"],
        y=data.data.loc[fit_mask, "x1_standardized"],
        w=data.data.loc[fit_mask, "x1_standardized_ERR"],
        full=True,
        deg=3,
    )
    print("* with least-squares:")
    print("   *", linear_fit.convert())
    print("   *", quadratic_fit.convert())
    print("   *", cubic_fit.convert(), "\n")

    # Work with sim data
    ####
    BS21 = Fitres(Path("data/COMBINED_SIMS.FITRES"), alpha)
    BS21.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    BS21.calc_HR()

    G10 = Fitres(Path("data/G10_SIMDATA.FITRES"), alpha)
    G10.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    G10.calc_HR()

    C11 = Fitres(Path("data/C11_SIMDATA.FITRES"), alpha)
    C11.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    C11.calc_HR()

    # Plots
    ###
    for c_split in c_splits:
        print("## Sub-sample Demographics\n")
        print(f"Splitting at c={c_split}")
        data.slipt_on_c(c_split)
        # data.plot_hists("FITPROB", f"fitprob_dist_{c_split}.pdf")  # Not in paper
        data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
        data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")
        data.plot_hists("zHD", f"redshift_dist_{c_split}.pdf")
        print("")

    # data.plot_fitprob_c()   # Not in paper
    data.plot_fitprob_binned_c()

    data.plot_hist("c", f"c_dist.pdf")
    # data.plot_hist_c_special("c", f"c_dist_special.pdf")   # Not in paper

    print("## Data vs Sims\n")
    plot_binned(
        data.data.loc[data.data["HOST_LOGMASS"] > 10],
        BS21.data.loc[BS21.data["HOST_LOGMASS"] > 10],
        x_col="c",
        y_col="x1_standardized",
        filename="color-luminosity-high_mass.png",
        fig_options={
            "data_name": "High Mass Hosts",
            "sim_name": "BS21",
            # TODO: define this elsewhere rather than copy and paste.
            # "y_label": "mB - mu(z) - 0.15 * x1",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
    plot_binned(
        data.data.loc[data.data["HOST_LOGMASS"] <= 10],
        BS21.data.loc[BS21.data["HOST_LOGMASS"] <= 10],
        x_col="c",
        y_col="x1_standardized",
        filename="color-luminosity-low_mass.png",
        fig_options={
            "data_name": "Low Mass Hosts",
            "sim_name": "BS21",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
    plot_binned(
        data.data,
        # BS21.data,
        x_col="c",
        y_col="HOST_LOGMASS",
        filename="mass-color.png",
        fig_options={
            "sim_name": "BS21",
            "y_label": r"$\log(M_{*})$",
            "ylim": [6.5, 13.5],
        },
    )
    # # TODO: currently cutting x-axis of anything <-0.5. This is crazy for host_logmass
    # plot_binned(
    #     data.data,
    #     BS21.data,
    #     "HOST_LOGMASS",
    #     "c",
    #     show_data=False,
    #     filename="color-mass.png",   # Not in paper
    # )

    plot_binned(
        data.data,
        BS21.data,
        "c",
        "x1_standardized",
        fit=fit,
        filename="color-luminosity-BS21.png",
        fig_options={
            "sim_name": "BS21",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -23],  # [-14.5, -21.5] elsewhere
        },
    )
    plot_binned(
        data.data,
        G10.data,
        x_col="c",
        y_col="x1_standardized",
        filename="color-luminosity-G10.png",
        fig_options={
            "sim_name": "G10",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
    plot_binned(
        data.data,
        C11.data,
        x_col="c",
        y_col="x1_standardized",
        filename="color-luminosity-C11.png",
        fig_options={
            "sim_name": "C11",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
