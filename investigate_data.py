""" investigate_red_sn.py

Requires python 3.9
"""

from pathlib import Path

import numpy as np
from colorama import Fore, Style

from br_util.stats import robust_scatter

from fitres import Fitres
from figures import plot_binned, bhm_diagnostic_plots, posterior_corner
from model_fitting import (
    rv_full_range,
    rv_least_squares,
    rv_broken_linear_bayesian,
    rv_broken_linear_frequentist,
)
from util import parse_cli
import broken_linear


if __name__ == "__main__":
    cli = parse_cli()

    # GLOBAL DEFAULTS
    #################
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose

    RUN_LINMIX = cli.linmix
    C_MAX_FIT = cli.cmax  # this is currently defined in several locations.
    BINS = cli.bins

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
    data.clean_data(x1err_max, x1_max, cerr_max, c_min, deduplicate=True)
    data.calc_HR()

    print("## Demographics of Sample\n")
    print("### Number of Highly Reddened SN")
    print(Fore.BLUE)
    print(f"Total SN {data.data.shape[0]}")
    print(f"There are {data.data.loc[data.data['c']>0.3, 'c'].shape[0]} SN with c>0.3")
    print(
        f"There are {data.data.loc[np.logical_and(data.data['c']>0.3, data.data['c']<=1.0), 'c'].shape[0]} SN with 0.3 < c <= 1.0"
    )
    print(
        f"There are {data.data.loc[data.data['c']>1.0, 'c'].shape[0]} SN with c > 1.0 \n"
    )

    data.slipt_on_c(0.95)
    print("### SN affected by c=1 boundary.")
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
        ].sort_values("c")
    )
    print(Style.RESET_ALL)

    print("## Basic Fit of Color-Luminosity Relationship")
    fit_mask = data.data["c"] <= C_MAX_FIT
    if RUN_LINMIX:
        fit = rv_full_range(data, fit_mask)
    else:
        # lm = None
        fit = None

    linear_fit = rv_least_squares(data, fit_mask)
    print("* with least-squares:")
    print("   *", linear_fit)
    # print("   *", quadratic_fit.convert())
    # print("   *", cucbic_fit.convert())
    print("")

    fitted = rv_broken_linear_bayesian(
        data.data.loc[fit_mask, ["c", "cERR", "x1_standardized", "x1_standardized_ERR"]]
    )
    # print(f'beta_pymc3 = {fitted.posterior["c"].mean(dim=("chain", "draw"))}')  # for bambi
    print("Medians:", fitted.posterior.median())
    print("StD:", fitted.posterior.std())
    print(
        Fore.BLUE
        + f'\nbeta (pymc3) = {np.tan(fitted.posterior["theta_cosmo"].median().values):.3f}'
        + " +/- "
        + f'{np.tan(robust_scatter(fitted.posterior["theta_cosmo"].values)):.3f}'
    )
    print(
        Fore.BLUE
        + "delta_beta (pymc3) = {:.3f} +/- {:.3f}".format(
            np.tan(fitted.posterior["delta_theta"].median().values),
            np.tan(robust_scatter(fitted.posterior["delta_theta"].values)),
        )
        + Style.RESET_ALL
    )
    bhm_diagnostic_plots(
        fitted,
        var_names=[
            "M0",
            "delta_theta",
            "theta_cosmo",
            "c_pop_mean",
            "c_pop_sigma",
            "c_pop_skewness",
            "sigma_int",
        ],
    )
    delta_rv_fit_freq = rv_broken_linear_frequentist(data, fit_mask)
    print(
        "Reduced Chi-square:",
        broken_linear.chi_square_params2slopes(delta_rv_fit_freq.x),
    )

    # Work with sim data
    ####
    BS21 = Fitres(Path("data/COMBINED_SIMS.FITRES"), alpha)
    BS21.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    BS21.calc_HR()
    print(
        f'Color quantiles for BS21: \n{BS21.data["c"].quantile([0.997, 0.999937, 0.99999943, 1.0])}'
    )

    G10 = Fitres(Path("data/G10_SIMDATA.FITRES"), alpha)
    G10.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    G10.calc_HR()

    C11 = Fitres(Path("data/C11_SIMDATA.FITRES"), alpha)
    C11.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    C11.calc_HR()

    M11 = Fitres(Path("data/MANDEL_SIMDATA.FITRES"), alpha)
    M11.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    M11.calc_HR()

    # Plots
    ###
    for c_split in c_splits:
        print("## Sub-sample Demographics\n")
        print(f"Splitting at c={c_split}")
        data.slipt_on_c(c_split)
        data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
        data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")
        data.plot_hists("zHD", f"redshift_dist_{c_split}.pdf")
        print("")

    data.plot_fitprob_binned_c()

    data.plot_hist("c", f"c_dist.pdf")

    posterior_corner(
        fitted.posterior,
        var_names=[
            "M0",
            "theta_cosmo",
            "delta_theta",
            "sigma_int",
            "c_pop_mean",
            "c_pop_sigma",
            "c_pop_skewness",
        ],
        filename="corner.pdf",
    )

    print("## Data vs Sims\n")
    plot_binned(
        data.data.loc[data.data["HOST_LOGMASS"] > 10],
        BS21.data.loc[BS21.data["HOST_LOGMASS"] > 10],
        x_col="c",
        y_col="x1_standardized",
        bins=BINS,
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
        bins=BINS,
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
        bins=BINS,
        filename="mass-color.png",
        fig_options={
            "sim_name": "BS21",
            "y_label": r"$\log(M_{*})$",
            "ylim": [6.5, 13.5],
        },
    )

    plot_binned(
        data.data,
        BS21.data,
        "c",
        "x1_standardized",
        fit=fitted.posterior.stack(draws=("chain", "draw")),
        c_max_fit=C_MAX_FIT,
        bins=BINS,
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
        bins=BINS,
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
        bins=BINS,
        filename="color-luminosity-C11.png",
        fig_options={
            "sim_name": "C11",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
    plot_binned(
        data.data,
        M11.data,
        x_col="c",
        y_col="x1_standardized",
        bins=BINS,
        filename="color-luminosity-M11.png",
        fig_options={
            "sim_name": "M11",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
    plot_binned(
        M11.data,
        BS21.data,
        x_col="c",
        y_col="x1_standardized",
        bins=BINS,
        filename="color-luminosity-M11-BS21.png",
        fig_options={
            "data_name": "M11",
            "sim_name": "BS21",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -21.5],
        },
    )
