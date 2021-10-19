""" investigate_red_sn.py

Requires python 3.9
"""

from pathlib import Path

import numpy as np

from fitres import Fitres
from figures import plot_binned
from model_fitting import rv_full_range, rv_least_squares
from util import parse_cli


if __name__ == "__main__":
    cli = parse_cli()

    # GLOBAL DEFAULTS
    #################
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose

    RUN_LINMIX = cli.linmix
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
        fit = rv_full_range(data, fit_mask)
    else:
        # lm = None
        fit = None

    linear_fit = rv_least_squares(data, fit_mask)
    print("* with least-squares:")
    print("   *", linear_fit)
    # print("   *", quadratic_fit.convert())
    # print("   *", cubic_fit.convert())
    print("")

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
        data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
        data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")
        data.plot_hists("zHD", f"redshift_dist_{c_split}.pdf")
        print("")

    data.plot_fitprob_binned_c()

    data.plot_hist("c", f"c_dist.pdf")

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
