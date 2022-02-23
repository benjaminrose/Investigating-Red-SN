"""Utility functions"""

__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction

from scipy.stats import binned_statistic

from br_util.stats import robust_scatter


def bin_dataset(data, x, y, bins=25, error_stat=robust_scatter):
    """
    Parameters
    -----------
    data: pandas.DataFrame
    x: str
        column for x-axis (binning-axis)
    y: str
        column name for y-axis (statistic axis)
    bins: int or sequence of scalars
        passed to scipy.stats.binned_statistic. Defaults to 25 bins
        (Not the same as scipy's default.)
    stat: str, function
        Passed to scipy.stats.binned_statistic. Defaults to
        `br_util.stats.robust_scatter`.
    """
    SCATTER_FLOOR = 0.1
    data = data.dropna(subset=[x, y])
    data_x_axis = data[x]
    data_y_axis = data[y]

    data_stat, data_edges, _ = binned_statistic(
        data_x_axis, data_y_axis, statistic="median", bins=bins
    )
    data_error, _, _ = binned_statistic(
        data_x_axis, data_y_axis, statistic=error_stat, bins=bins
    )
    data_error[data_error == 0] = SCATTER_FLOOR
    return data_x_axis, data_y_axis, data_stat, data_edges, data_error


def parse_cli():
    arg_parser = ArgumentParser(description=__doc__)
    arg_parser.add_argument("--version", action="version", version=__version__)
    arg_parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="number of bins to use in color-luminosity plot (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--cmax",
        type=float,
        default=1.3244,  # max in BS21 on Oct 22, 2021
        help="maximum c used in fitting BETA (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--linmix",
        action=BooleanOptionalAction,
        default=False,
        help="run LINMIX to fit for BETA",
    )  # python 3.9
    arg_parser.add_argument(
        "--fast",
        action="store_true",
        help="skips part of the analysis to increase speed, forces `--no-linmix` (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--sims",
        action=BooleanOptionalAction,
        default=False,
        help="run broken-linear on sims, then exit",
    )  # python 3.9
    arg_parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="used for light-curve shape standardization (default: %(default)s)",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="turn on verbose output (default: %(default)s)",
    )

    return arg_parser.parse_args()


def checking_sims():
    from astropy.cosmology import wCDM

    COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)
    G10.data["SIM_x1_stand"] = (
        G10.data["SIM_mB"]
        - COSMO.distmod(G10.data["SIM_ZCMB"]).value
        - G10.data["SIM_alpha"] * G10.data["SIM_x1"]
    )
    G10.data["delta_m"] = G10.data["x1_standardized"] - G10.data["SIM_x1_stand"]
    G10.data["delta_c"] = G10.data["c"] - G10.data["SIM_c"]
    print(
        "Is M' correct?\n",
        G10.data[["delta_m", "IDSURVEY"]].nlargest(
            n=50, columns="delta_m"
        ),  # .describe(),
    )
    print(
        "SN with largest delta c (obs - sim)\n",
        G10.data[["delta_c", "IDSURVEY"]].nlargest(
            n=50, columns="delta_c"
        ),  # .describe(),
    )
    import seaborn as sns

    sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)
    ax = sns.histplot(
        data=G10.data["delta_m"],
        color="b",
        cumulative=False,
        bins="auto",
        element="step",
        fill=False,
    )
    sns.rugplot(
        data=G10.data["delta_m"],
        height=-0.02,
        lw=1,
        clip_on=False,
        ax=ax,
    )
    from br_util.plot import save_plot, new_figure
    import matplotlib.pyplot as plt

    save_plot("del.pdf")

    new_figure()
    plt.plot(G10.data["SIM_mB"], G10.data["SIM_mB"] - G10.data["SIM_DLMAG"], ".")
    plt.xlabel("SIM_mB")
    plt.ylabel("SIM_mB - SIM_DLMAG")
    save_plot("SIM_DLMAG-SIM_mB.pdf")

    new_figure()
    plt.plot(G10.data["SIM_mB"], G10.data["delta_m"], ".")
    plt.xlabel("SIM_mB")
    plt.ylabel("delta_m")
    save_plot("delta_m-SIM_mB.pdf")

    new_figure()
    plt.plot(G10.data["mB"], G10.data["delta_m"], ".")
    plt.xlabel("mB")
    plt.ylabel("delta_m")
    save_plot("delta_m-mB.pdf")

    new_figure()
    plt.plot(G10.data["mB"] - G10.data["SIM_mB"], G10.data["delta_m"], ".")
    plt.xlabel("mB - SIM_mB")
    plt.ylabel("delta_m")
    save_plot("delta_m-mB-SIM_mB.pdf")

    new_figure()
    plt.plot(G10.data["SIM_x1"], G10.data["delta_m"], ".")
    plt.xlabel("SIM_x1")
    plt.ylabel("delta_m")
    save_plot("delta_m-SIM_x1.pdf")

    new_figure()
    plt.plot(G10.data["x1"], G10.data["delta_m"], ".")
    plt.xlabel("x1")
    plt.ylabel("delta_m")
    save_plot("delta_m-x1.pdf")

    new_figure()
    plt.plot(G10.data["x1"] - G10.data["SIM_x1"], G10.data["delta_m"], ".")
    plt.xlabel("x1 - SIM_x1")
    plt.ylabel("delta_m")
    save_plot("delta_m-x1-SIM_x1.pdf")

    new_figure()
    plt.plot(G10.data["x1ERR"], G10.data["delta_m"], ".")
    plt.xlabel("x1ERR")
    plt.ylabel("delta_m")
    save_plot("delta_m-x1ERR.pdf")

    new_figure()
    plt.plot(G10.data["SIM_c"], G10.data["delta_m"], ".")
    plt.xlabel("SIM_c")
    plt.ylabel("delta_m")
    save_plot("delta_m-SIM_c.pdf")

    new_figure()
    plt.plot(G10.data["c"], G10.data["delta_m"], ".")
    plt.xlabel("c")
    plt.ylabel("delta_m")
    save_plot("delta_m-c.pdf")

    new_figure()
    plt.plot(G10.data["c"] - G10.data["SIM_c"], G10.data["delta_m"], ".")
    plt.xlabel("c - SIM_c")
    plt.ylabel("delta_m")
    save_plot("delta_m-c-SIM_c.pdf")

    new_figure()
    plt.plot(G10.data["zHD"], G10.data["c"] - G10.data["SIM_c"], ".")
    plt.xlabel("zHD")
    plt.ylabel("c - SIM_c")
    save_plot("zHD-c-SIM_c.pdf")

    new_figure()
    plt.plot(G10.data["c"], G10.data["c"] - G10.data["SIM_c"], ".")
    plt.xlabel("c")
    plt.ylabel("c - SIM_c")
    save_plot("c-SIM_c-c.pdf")

    new_figure()
    plt.plot(G10.data["FITPROB"], G10.data["c"] - G10.data["SIM_c"], ".")
    plt.xlabel("FITPROB")
    plt.ylabel("c - SIM_c")
    save_plot("FITPROB-SIM_c-c.pdf")

    new_figure()
    plt.plot(G10.data["SIM_c"], G10.data["c"], ".")
    plt.xlabel("SIM_c")
    plt.ylabel("c")
    save_plot("SIM_c-c-g10.pdf")

    new_figure()
    plt.plot(BS21.data["SIM_c"], BS21.data["c"], ".")
    plt.xlabel("SIM_c")
    plt.ylabel("c")
    save_plot("SIM_c-c-bs21.pdf")

    new_figure()
    plt.plot(G10.data["SIM_c"], G10.data["c"] - G10.data["SIM_c"], ".")
    plt.xlabel("SIM_c")
    plt.ylabel("c - SIM_c")
    save_plot("SIM_c-SIM_c-c.pdf")

    new_figure()
    plt.plot(G10.data["cERR"], G10.data["delta_m"], ".")
    plt.xlabel("cERR")
    plt.ylabel("delta_m")
    save_plot("delta_m-cERR.pdf")

    new_figure()
    plt.plot(G10.data["SIM_ZCMB"], G10.data["delta_m"], ".")
    plt.xlabel("SIM_ZCMB")
    plt.ylabel("delta_m")
    save_plot("delta_m-SIM_ZCMB.pdf")

    new_figure()
    plt.plot(G10.data["zHD"], G10.data["delta_m"], ".")
    plt.xlabel("zHD")
    plt.ylabel("delta_m")
    save_plot("delta_m-zHD.pdf")

    new_figure()
    plt.plot(G10.data["SIM_x1_stand"], G10.data["delta_m"], ".")
    plt.xlabel("SIM_x1_stand")
    plt.ylabel("delta_m")
    save_plot("delta_m-SIM_x1_stand.pdf")

    new_figure()
    plt.plot(G10.data["x1_standardized"], G10.data["delta_m"], ".")
    plt.xlabel("x1_standardized")
    plt.ylabel("delta_m")
    save_plot("delta_m-x1_stand.pdf")

    print(
        G10.data[G10.data["delta_m"] > 1][
            [
                "SIM_mB",
                "mB",
                "SIM_x1",
                "x1",
                "x1ERR",
                "SIM_c",
                "c",
                "cERR",
                "SIM_ZCMB",
                "zHD",
                "SIM_x1_stand",
                "x1_standardized",
                "SIM_alpha",
                "SIM_beta",
                # "SIM_RV",
                # "MWEBV",
            ]
        ].describe()
    )

    G10.data["c"] = 1.5 * G10.data["SIM_c"] + 0.15
    # G10.data["x1_standardized"] = (
    #     G10.data["SIM_mB"]
    #     - COSMO.distmod(G10.data["SIM_ZCMB"]).value
    #     - G10.data["SIM_alpha"] * G10.data["SIM_x1"]
    # )
    G10.data["x1_standardized"] = 3.1 * G10.data["c"] - 19.4 + 0.1 * np.random.randn(1)
    G10.data["x1_standardized_ERR"] = 0.1
    # rv_full_range(G10, G10.data["SIM_c"] <= 0.3, comment=" G10 cosmo, sim values")
    # fitted = rv_broken_linear_bayesian(
    #     G10.data.loc[
    #         G10.data["SIM_c"] <= 1.0,
    #         ["c", "cERR", "x1_standardized", "x1_standardized_ERR"],
    #     ]
    # )
    # print(
    #     Fore.BLUE
    #     + f'\nbeta (pymc3) = {np.median(np.tan(fitted.posterior["θ"].values)):.3f}'
    #     + " +/- "
    #     + f'{robust_scatter(np.tan(fitted.posterior["θ"].values)):.3f}'
    # )
    # print(
    #     Fore.BLUE
    #     + "delta_beta (pymc3) = {:.3f} +/- {:.3f}".format(
    #         np.median(np.tan(fitted.posterior["Δ_θ"].values)),
    #         robust_scatter(np.tan(fitted.posterior["Δ_θ"].values)),
    #     )
    #     + Style.RESET_ALL
    # )
    fit_mask = data.data["c"] <= C_MAX_FIT
    fitted = rv_broken_linear_bayesian(
        data.data.loc[fit_mask, ["c", "cERR", "x1_standardized", "x1_standardized_ERR"]]
    )
    # print(f'beta_pymc3 = {fitted.posterior["c"].mean(dim=("chain", "draw"))}')  # for bambi
    print("Medians:", fitted.posterior.median())
    print("StD:", fitted.posterior.std())
    bhm_diagnostic_plots(
        fitted,
        var_names=[
            "M_0",
            "Δ_θ",
            "θ",
            "μ_c",
            "σ_c",
            "α_c",
            "σ",
        ],
    )
    linear_fit = rv_least_squares(G10, G10.data["SIM_c"] <= 0.3)
    print(linear_fit)
