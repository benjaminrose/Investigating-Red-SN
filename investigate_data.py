""" investigate_red_sn.py

Requires python 3.9
"""

from pathlib import Path

import numpy as np
from colorama import Fore, Style

from br_util.stats import robust_scatter

from fitres import Fitres
from figures import (
    plot_binned,
    bhm_diagnostic_plots,
    posterior_corner,
    chi2_bin,
    plot_rms_c,
    chi_color_min,
)
from model_fitting import (
    rv_full_range,
    rv_least_squares,
    rv_broken_linear_bayesian,
    rv_broken_linear_frequentist,
)
from util import parse_cli, calc_chi2, kr_sims
import broken_linear


if __name__ == "__main__":
    cli = parse_cli()

    # GLOBAL DEFAULTS
    #################
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose

    FAST = cli.fast
    ALL = cli.all
    TALK = cli.talk
    if FAST:
        RUN_LINMIX = False
    else:
        RUN_LINMIX = cli.linmix
    ON_SIMS = cli.sims
    C_MAX_FIT = cli.cmax  # this is currently defined in several locations.
    BINS = cli.bins
    RMS_SIMS = True
    BREAK_VALUE = cli.break_value
    Z_MIN = cli.zmin

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

    data = Fitres(data_file, Z_MIN, alpha, VERBOSE)
    # Interesting (for whatever reason) SN that get lost in data cleaning
    # only look at this if no cuts have been applied.
    if Z_MIN <= 0.0001:
        print("## objects that get cut")
        get_cut = [
            "1999ek",
            "2008fp",
            "2000E",
            "2006X",
        ]
        print(
            data.data.loc[
                get_cut,
                [
                    "CIDint",
                    "IDSURVEY",
                    "zHD",
                    "x1",
                    "x1ERR",
                    "c",
                    "cERR",
                    "PKMJDERR",
                    "TrestMIN",
                    "HOST_LOGMASS",
                    "MWEBV",
                ],
            ].sort_values("c")
        )
    data.clean_data(x1err_max, x1_max, cerr_max, c_min, deduplicate=True)
    data.calc_HR()

    print("## high redshift hihgly reddened SN")
    is_z_gt_04 = np.logical_and(data.data["zHD"] > 0.2, data.data["c"] > 0.3)
    print(
        data.data.loc[
            is_z_gt_04,
            [
                "CIDint",
                "IDSURVEY",
                "zHD",
                "x1",
                "x1ERR",
                "c",
                "cERR",
                "PKMJDERR",
            ],
        ].sort_values("c")
    )

    HST = False
    if HST:
        # Y(F105W) at z=0.05
        data.data.loc[
            np.logical_and(0.035 < data.data["zHD"], data.data["zHD"] < 0.07),
            [
                "IDSURVEY",
                "zHD",
                "RA",
                "DEC",
                "HOST_LOGMASS",
                "HOSTGAL_SFR",
                "HOST_sSFR",
                "HOST_RA",
                "HOST_DEC",
            ],
        ].sort_values(["DEC", "RA", "zHD"]).to_csv("z0.05_hosts.csv", na_rep="-999")
        # J(F125W) at z=0.2
        data.data.loc[
            np.logical_and(0.18 < data.data["zHD"], data.data["zHD"] < 0.22),
            [
                "IDSURVEY",
                "zHD",
                "RA",
                "DEC",
                "HOST_LOGMASS",
                "HOSTGAL_SFR",
                "HOST_sSFR",
                "HOST_RA",
                "HOST_DEC",
            ],
        ].sort_values(["DEC", "RA", "zHD"]).to_csv("z0.2_hosts.csv", na_rep="-999")
        import sys

        sys.exit()

    # Import sim data
    ####
    BS21 = Fitres(Path("data/COMBINED_SIMS.FITRES"), alpha)
    BS21.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    BS21.calc_HR()
    print(
        f'Color quantiles for BS21: \n{BS21.data["c"].quantile([0.997, 0.999937, 0.99999943, 1.0])}'
    )
    if ON_SIMS:
        fit_mask = BS21.data["c"] <= C_MAX_FIT
        fitted = rv_broken_linear_bayesian(
            BS21.data.loc[
                fit_mask, ["c", "cERR", "x1_standardized", "x1_standardized_ERR"]
            ],
            BREAK_VALUE,
            FAST,
        )
        print("Running on BS21 sims.")
        print("Medians:", fitted.posterior.median())
        print("StD:", fitted.posterior.std())

        import sys

        sys.exit()

    G10 = Fitres(Path("data/G10_SIMDATA.FITRES"), Z_MIN, alpha)
    G10.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    G10.calc_HR()

    C11 = Fitres(Path("data/C11_SIMDATA.FITRES"), Z_MIN, alpha)
    C11.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    C11.calc_HR()

    M11 = Fitres(Path("data/MANDEL_SIMDATA.FITRES"), Z_MIN, alpha)
    M11.clean_data(x1err_max, x1_max, cerr_max, c_min, sim=True)
    M11.calc_HR()

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

    data.slipt_on_c(BREAK_VALUE)
    print("### mean color values")
    print(f"Full sample mean: {data.data['c'].describe()}")
    print(f"Cosmo sample mean: {data.blue_subsample['c'].describe()}")

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
    if Z_MIN <= 0.0001:
        print("### Possible Iax Outlier")
        print(
            data.data.loc[
                np.bitwise_and(
                    data.data["c"] < 1.0, data.data["x1_standardized"] > -16
                ),
                [
                    # "zHD",
                    "x1",
                    "x1ERR",
                    "c",
                    "cERR",
                    "x1_standardized",
                    # "x1_standardized_ERR",
                    "FITPROB",
                ],
            ]
        )
        print("### Possible New H0 Calibrators.")
        print(
            data.data.loc[
                # Not in cosmo sample, and have mu<34 (z~0.015)
                np.bitwise_and(data.data["c"] > 0.3, data.data["zHD"] < 0.015),
                [
                    "CIDint",
                    "IDSURVEY",
                    "zHD",
                    "c",
                    "cERR",
                ],
            ].sort_values("zHD")
        )
        print("### Delta Mag, assuming: M0=-19.32 & beta=3.10.")
        objects_with_rv = [
            "1989B",
            "1998bu",
            "1999cl",
            "1999cp",
            "1999ee",
            # "1999ek",  #
            # "1999pg",  #
            # "2000E",  #
            "2000bh",
            "2000ca",
            "2002bo",
            # "2002cv",  #
            # "2003cg",  #
            "2006X",  #
            "2008fp",
            "2012cg",
            "2014J",
        ]
        print(
            data.data.loc[
                objects_with_rv,
                [
                    "CIDint",
                    "IDSURVEY",
                    "zHD",
                    "c",
                    "cERR",
                    "HOST_LOGMASS",
                ],
            ]
        )
        print(
            "delta-mag for c>0.3 (beta=3.01, delta-beta=-0.07):",
            data.data.loc[
                objects_with_rv,
                ["x1_standardized"],
            ]
            - (
                (3.00956967 - 0.06739) * data.data.loc[objects_with_rv, ["c"]].values
                - 19.32
                - 0.3 * 3.00956967
            ),
        )
    print(Style.RESET_ALL)

    print("## Basic Fit of Color-Luminosity Relationship")
    fit_mask = data.data["c"] <= C_MAX_FIT
    if RUN_LINMIX:
        fit = rv_full_range(data, fit_mask)
        # rv_full_range(data, data.data["c"] <= 0.3, comment=" cosmo")
        # rv_full_range(G10, G10.data["c"] <= 0.3, comment=" G10 cosmo")
    else:
        # lm = None
        fit = None

    if ALL:
        linear_fit = rv_least_squares(data, fit_mask)
        print("* with least-squares:")
        print("   *", linear_fit)
        print("")

    fitted = rv_broken_linear_bayesian(
        data.data.loc[
            fit_mask, ["c", "cERR", "x1_standardized", "x1_standardized_ERR"]
        ],
        BREAK_VALUE,
        FAST,
    )
    print("Medians:", fitted.posterior.median())
    print("StD:", fitted.posterior.std())
    print(
        Fore.BLUE
        + f'\nbeta (pymc3) = {np.median(np.tan(fitted.posterior["??"].values)):.3f}'
        + " +/- "
        + f'{robust_scatter(np.tan(fitted.posterior["??"].values)):.3f}'
    )
    print(
        Fore.BLUE
        + "delta_beta (pymc3) = {:.3f} +/- {:.3f}".format(
            np.median(np.tan(fitted.posterior["??_??"].values)),
            robust_scatter(np.tan(fitted.posterior["??_??"].values)),
        )
        + Style.RESET_ALL
    )
    if not FAST:
        var_names = [
            "M'_0",
            "??_??",
            "??",
            "??_c",
            "??_c",
            "??_c",
            "??",
        ]
        bhm_diagnostic_plots(fitted, var_names=var_names)
    if ALL:
        delta_rv_fit_freq = rv_broken_linear_frequentist(data, fit_mask)
        print(
            "Reduced Chi-square:",
            broken_linear.chi_square_params2slopes(delta_rv_fit_freq.x),
        )

    # Plots
    ###
    if not FAST:
        for c_split in c_splits:
            print("## Sub-sample Demographics\n")
            print(f"Splitting at c={c_split}")
            data.slipt_on_c(c_split)
            data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
            data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")
            data.plot_hists("zHD", f"redshift_dist_{c_split}.pdf")
            print("")
        if ALL:
            data.plot_fitprob_binned_c()

    data.plot_hist_plus("c", [G10.data, C11.data, M11.data, BS21.data], f"c_dist.pdf")

    print(
        f"Number of objects per c bin (of size 0.1); {np.histogram(data.data['c'], bins=np.arange(-0.3, 2.0, 0.1))[0]}"
    )

    if TALK:
        data.plot_hist("c", f"c_dist_talk.pdf")

    if not FAST:
        posterior_corner(
            fitted.posterior,
            var_names=[
                "M'_0",
                "??",
                "??_??",
                "??",
                "??_c",
                "??_c",
                "??_c",
            ],
            filename="corner.pdf",
        )

    luminosity_sims = kr_sims(
        G10.data, C11.data, M11.data, BS21.data, c_break=BREAK_VALUE
    )

    print("## Data vs Sims\n")
    if ALL:
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
                "sim_name": "P22",
                "y_label": r"M$'$ (mag)",
                "y_flip": True,
                "ylim": [-14.5, -21.5],
            },
        )
    if not FAST:
        plot_binned(
            data.data,
            # BS21.data,
            x_col="c",
            y_col="HOST_LOGMASS",
            bins=BINS,
            filename="mass-color.pdf",
            fig_options={
                "sim_name": "P22",
                "y_label": r"$\log(M_{*}/M_{\odot})$",
                "ylim": [6.5, 13.5],
            },
        )

    if RMS_SIMS:
        scatter_sims = kr_sims(
            G10.data,
            C11.data,
            M11.data,
            BS21.data,
            c_break=BREAK_VALUE,
            var_name="",
            fit=fitted.posterior.stack(draws=("chain", "draw")),
        )
    else:
        scatter_sims = None
    scatter_data = plot_rms_c(
        data.data,
        fit=fitted.posterior.stack(draws=("chain", "draw")),
        c_max_fit=C_MAX_FIT,
        c_break=BREAK_VALUE,
        bins=BINS,
        scatter_sims=scatter_sims,
    )

    if TALK:
        plot_binned(
            data.data,
            x_col="c",
            y_col="x1_standardized",
            fit=fitted.posterior.stack(draws=("chain", "draw")),
            c_max_fit=C_MAX_FIT,
            c_break=BREAK_VALUE,
            bins=BINS,
            scatter=scatter_data,
            filename="color-luminosity-talk.pdf",
            fig_options={
                "y_label": r"M$'$ (mag)",
                "y_flip": True,
                "ylim": [-14.5, -20.9],  # [-14.5, -21.5] elsewhere
            },
        )
    plot_binned(
        data.data,
        # BS21.data,
        x_col="c",
        y_col="x1_standardized",
        fit=fitted.posterior.stack(draws=("chain", "draw")),
        c_max_fit=C_MAX_FIT,
        c_break=BREAK_VALUE,
        bins=BINS,
        model_krs=luminosity_sims,
        scatter=scatter_data,
        add_14J=True,
        filename="color-luminosity-BS21.pdf",
        fig_options={
            "sim_name": "P22",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
            "ylim": [-14.5, -22.5],  # [-14.5, -21.5] elsewhere
        },
    )
    if ALL:
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
            filename="color-luminosity-M11-BS21.pdf",
            fig_options={
                "data_name": "M11",
                "sim_name": "P22",
                "y_label": r"M$'$ (mag)",
                "y_flip": True,
                "ylim": [-14.5, -21.5],
            },
        )
        chi2_bin(data.data, G10.data, C11.data, M11.data, BS21.data)
        chi_color_min(data.data, G10.data, C11.data, M11.data, BS21.data)

    print("full data sets")
    calc_chi2(
        data.data,
        G10.data,
        C11.data,
        M11.data,
        BS21.data,
        scatter_data,
        luminosity_sims,
    )
    print("\nc<1")
    calc_chi2(
        data.data.loc[data.data["c"] < 1.0],
        G10.data.loc[G10.data["c"] < 1.0],
        C11.data.loc[C11.data["c"] < 1.0],
        M11.data.loc[M11.data["c"] < 1.0],
        BS21.data.loc[BS21.data["c"] < 1.0],
        scatter_data,
        luminosity_sims,
    )
    print("\nc<0.5")
    calc_chi2(
        data.data.loc[data.data["c"] < 0.5],
        G10.data.loc[G10.data["c"] < 0.5],
        C11.data.loc[C11.data["c"] < 0.5],
        M11.data.loc[M11.data["c"] < 0.5],
        BS21.data.loc[BS21.data["c"] < 0.5],
        scatter_data,
        luminosity_sims,
    )
    print("\nc<0.3")
    calc_chi2(
        data.data.loc[data.data["c"] < 0.3],
        G10.data.loc[G10.data["c"] < 0.3],
        C11.data.loc[C11.data["c"] < 0.3],
        M11.data.loc[M11.data["c"] < 0.3],
        BS21.data.loc[BS21.data["c"] < 0.3],
        scatter_data,
        luminosity_sims,
    )
