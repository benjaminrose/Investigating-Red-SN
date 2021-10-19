"""Module containing my figure creating functions
"""
import seaborn as sns
import numpy as np

from br_util.plot import save_plot, new_figure
from br_util.stats import robust_scatter

from util import bin_dataset
from hypothesis_testing import ks2d

__all__ = ["plot_binned"]

sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)
USE_KS2D = False
C_MAX_FIT = 2.0


def plot_binned(
    data,
    sim=None,
    x_col="c",
    y_col="HOST_LOGMASS",
    fit=None,
    show_data=True,
    # split_mass=False,
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

    # if split_mass:
    #     data_high = data.loc[data["HOST_LOGMASS"] > 10]
    #     data_low = data.loc[data["HOST_LOGMASS"] <= 10]
    #     if sim is not None:
    #         sim_high = sim.loc[sim["HOST_LOGMASS"] > 10]
    #         sim_low = sim.loc[sim["HOST_LOGMASS"] <= 10]

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
        _add_fit(ax, data_x, fit)

    _add_fig_options(ax, x_col, y_col, fig_options)

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
        _data_v_sim(
            data_x,
            data_y,
            data_stat,
            data_error,
            sim_x,
            sim_y,
            sim_stat,
            sim_error,
            bins,
            filename,
            fig_options,
        )


def _add_fit(ax, data_x, fit):
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
    # ax.plot(
    #     xs,
    #     np.median(lm_cosmo.chain["alpha"]) + xs * np.median(lm_cosmo.chain["beta"]),
    #     label=r"$\beta_{cosmo}=$"
    #     + f"{np.median(lm_cosmo.chain['beta']):.2f}"
    #     + r" $\pm$ "
    #     + f"{robust_scatter(lm_cosmo.chain['beta']):.2f}",
    # )
    # ax.plot(
    #     xs,
    #     np.median(lm_red.chain["alpha"]) + xs * np.median(lm_red.chain["beta"]),
    #     label=r"$\beta_{red}=$"
    #     + f"{np.median(lm_red.chain['beta']):.2f}"
    #     + r" $\pm$ "
    #     + f"{robust_scatter(lm_red.chain['beta']):.2f}",
    # )
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


def _add_fig_options(ax, x_col, y_col, fig_options):
    if fig_options.get("y_flip", False):
        ax.invert_yaxis()
    ax.set_xlabel(fig_options.get("x_label", x_col))
    ax.set_ylabel(fig_options.get("y_label", y_col))
    ax.set_ylim(fig_options.get("ylim", ax.get_ylim()))


def _data_v_sim(
    data_x,
    data_y,
    data_stat,
    data_error,
    sim_x,
    sim_y,
    sim_stat,
    sim_error,
    bins,
    filename,
    fig_options,
):
    ks_p, data_step, sim_step = ks2d(
        data_x, data_y, sim_x, sim_y, fig_options, USE_KS2D
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
