""" investigate_red_sn.py

Requires python 3.9
"""
__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import binned_statistic, ks_2samp
import seaborn as sns

import linmix

from br_util.stats import robust_scatter
from br_util.plot import save_plot, new_figure

from KS2D.KS2D import ks2d2s

from fitres import Fitres
from figures import *

sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)


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


def plot_binned(
    data,
    sim=None,
    x_col="c",
    y_col="HOST_LOGMASS",
    fit=None,
    show_data=True,
    split_mass=False,
    filename="",
    fig_options={},
):
    """plot binned values of data.

    Parameters
    ----------
    data : pandas.DataFrame
    sim=None : pandas.DataFrame
        If None, then ignored.
    x_col : str ("c")
        DataFrame column name to use for x-axis (axis the bins are applied).
    y_col : str (""HOST_LOGMASS")
        DataFrame column name to use for y-axis (axis the summary statistic is applied).
    fit : linmix.LinMix.chain (None)
        The linear fit. Assumes it is from `linmix`. Does not plot anything value is None.
    show_data: bool (True)
        If True, show data
    split_mass: bool (False)
        Plot data/sims as high and low host stellar mass seperately.
    filename: str ("")
        File name to use when saving figure. If left as the empty string,
        figure is shown but not saved.
    fig_options : dict
        Contrains user overrides to be used in figure creation. Keys are
        "sim_name", "data_name", "x_label", "y_label", "y_flip", "y_lim", "leg_loc".
    """
    bins = 25

    if split_mass:
        data_high = data.loc[data["HOST_LOGMASS"] > 10]
        data_low = data.loc[data["HOST_LOGMASS"] <= 10]
        if sim is not None:
            sim_high = sim.loc[sim["HOST_LOGMASS"] > 10]
            sim_low = sim.loc[sim["HOST_LOGMASS"] <= 10]

    data_x, data_y, data_stat, data_edges, data_error = bin_dataset(
        data, x_col, y_col, bins=bins
    )
    if sim is not None:
        sim_x, sim_y, sim_stat, sim_edges, sim_error = bin_dataset(
            sim, x_col, y_col, bins=data_edges
        )

    _, ax = new_figure()

    if show_data:
        if sim is not None:
            ax.plot(
                sim_x,
                sim_y,
                ".",
                color="tab:green",
                markersize=3,
                alpha=0.1,
                label=fig_options.get("sim_name", "Simulation"),
            )
        ax.plot(
            data_x,
            data_y,
            ".",
            color="tab:blue",
            markersize=3,
            alpha=0.3,
            label=fig_options.get("data_name", "Pantheon+"),
        )
    if sim is not None:
        ax.errorbar(
            (sim_edges[:-1] + sim_edges[1:]) / 2,
            sim_stat,
            yerr=sim_error,
            color="tab:pink",
            fmt="^",
            label="Binned simulation",
        )
    ax.errorbar(
        (data_edges[:-1] + data_edges[1:]) / 2,
        data_stat,
        yerr=data_error,
        color="tab:orange",
        fmt=">",
        label="Binned data",
    )

    if fit is not None:
        # relevantly bring in global constant (from CLI) into local scope.
        if C_MAX_FIT > data_x.max():
            print(
                f"{C_MAX_FIT = } is above {data_x.max() = }. Plot of linear fit will not go past the data.",
                end="\n\n",
            )
            c_max_fit = data_x.max()
        else:
            c_max_fit = C_MAX_FIT

        xs = np.arange(data_x.min(), c_max_fit, 0.01)

        # only plot every 25 chains. For 2000 chains, this is 4%
        downsample_frac = 0.15  # 5--10% may be better for a long chain
        # // floors it but still keeps it as a float.
        for i in range(0, len(fit), round((len(fit) / downsample_frac) / len(fit))):
            ys = fit[i]["alpha"] + xs * fit[i]["beta"]
            ax.plot(xs, ys, color="0.5", alpha=0.02)
        ys = np.median(fit["alpha"]) + xs * np.median(fit["beta"])
        ax.plot(
            xs,
            ys,
            color="k",
            label=r"$\beta=$"
            + f"{np.median(fit['beta']):.2f}"
            + r" $\pm$ "
            + f"{robust_scatter(fit['beta']):.2f}",
        )
        ax.plot(
            xs,
            np.median(lm_cosmo.chain["alpha"]) + xs * np.median(lm_cosmo.chain["beta"]),
            label=r"$\beta_{cosmo}=$"
            + f"{np.median(lm_cosmo.chain['beta']):.2f}"
            + r" $\pm$ "
            + f"{robust_scatter(lm_cosmo.chain['beta']):.2f}",
        )
        ax.plot(
            xs,
            np.median(lm_red.chain["alpha"]) + xs * np.median(lm_red.chain["beta"]),
            label=r"$\beta_{red}=$"
            + f"{np.median(lm_red.chain['beta']):.2f}"
            + r" $\pm$ "
            + f"{robust_scatter(lm_red.chain['beta']):.2f}",
        )
        # ax.plot(
        #     xs,
        #     linear_fit.convert().coef[0] + xs * linear_fit.convert().coef[1],
        #     label="Linear Least-Squares",
        # )
        # ax.plot(
        #     xs,
        #     quadratic_fit.convert().coef[0]
        #     + xs * quadratic_fit.convert().coef[1]
        #     + xs ** 2 * quadratic_fit.convert().coef[2],
        #     label="Quadratic Least-Squares",
        # )

    if fig_options.get("y_flip", False):
        ax.invert_yaxis()
    ax.set_xlabel(fig_options.get("x_label", x_col))
    ax.set_ylabel(fig_options.get("y_label", y_col))
    ax.set_ylim(fig_options.get("ylim", ax.get_ylim()))

    if fit is not None:
        ncol = 2
    else:
        ncol = 1
    leg = ax.legend(
        fontsize="xx-small", loc=fig_options.get("leg_loc", "best"), ncol=ncol
    )
    for line in leg.get_lines():
        line._legmarker.set_alpha(1)
        line._legmarker.set_markersize(6)

    save_plot(filename)

    if sim is not None:
        # ks, p = ks_2samp(data_stat, sim_stat)
        print("Running KS2D ...")
        if USE_KS2D:
            data_step = 1
            sim_step = 1
        else:
            data_step = 20
            sim_step = 4950 if fig_options.get("sim_name") == "BS21" else 370
        _, ks_p = ks2d2s(
            np.stack((data_x[::data_step], data_y[::data_step]), axis=1),
            np.stack((sim_x[::sim_step], sim_y[::sim_step]), axis=1),
        )

        print(f"For {filename},")
        print(
            f"KS-test of binned stats: {ks_p:.3g}. With N=({len(data_x[::data_step])}, {len(sim_y[::sim_step])})."
        )
        chi_square = np.nansum(
            (data_stat - sim_stat) ** 2 / (sim_error ** 2 + data_error ** 2)
        )
        print(f"reduced chi-2 binned stats: {chi_square/bins:.3f}\n")
        # I want the Mann-Whitney-U of the data in each bin.
        # Then somehow compute a single value from the 25 Mann-Whitney-U values.


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


def parse_cli():
    arg_parser = ArgumentParser(description=__doc__)
    arg_parser.add_argument("--version", action="version", version=__version__)
    arg_parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="number of bins to use in color-luminosity plot (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--cmax",
        type=float,
        default=2.0,
        help="maximum c used in fitting BETA (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--linmix",
        action=BooleanOptionalAction,
        default=False,
        help="run LINMIX to fit for BETA",
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


if __name__ == "__main__":
    cli = parse_cli()

    # GLOBAL DEFAULTS
    #################
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose

    RUN_LINMIX = cli.linmix
    # defaults to 5000, 2000 is good for our our data set, 500 for fast
    LINMIX_MIN_ITR = 200
    C_MAX_FIT = cli.cmax
    USE_KS2D = False

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

        fit_mask_cosmo = (-0.3 <= data.data["c"]) & (data.data["c"] <= 0.3)
        lm_cosmo = linmix.LinMix(
            x=data.data.loc[fit_mask_cosmo, "c"],
            y=data.data.loc[fit_mask_cosmo, "x1_standardized"],
            xsig=data.data.loc[fit_mask_cosmo, "cERR"],
            ysig=data.data.loc[fit_mask_cosmo, "x1_standardized_ERR"],
        )
        lm_cosmo.run_mcmc(miniter=LINMIX_MIN_ITR)
        print(
            f"\n* with LINMIX: Beta_cosmo = {np.median(lm_cosmo.chain['beta']):.3f} +/- {robust_scatter(lm_cosmo.chain['beta']):.3f}"
        )

        fit_mask_red = 0.3 < data.data["c"]
        lm_red = linmix.LinMix(
            x=data.data.loc[fit_mask_red, "c"],
            y=data.data.loc[fit_mask_red, "x1_standardized"],
            xsig=data.data.loc[fit_mask_red, "cERR"],
            ysig=data.data.loc[fit_mask_red, "x1_standardized_ERR"],
        )
        lm_red.run_mcmc(miniter=LINMIX_MIN_ITR)
        print(
            f"\n* with LINMIX: Beta_red = {np.median(lm_red.chain['beta']):.3f} +/- {robust_scatter(lm_red.chain['beta']):.3f}"
        )
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
