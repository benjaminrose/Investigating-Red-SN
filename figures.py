"""Module containing my figure creating functions
"""
from collections import Counter

import seaborn as sns
from matplotlib.pyplot import savefig
import numpy as np
import pandas as pd
import arviz as az
import corner
from scipy.stats import binned_statistic

from br_util.plot import save_plot, new_figure
from br_util.stats import robust_scatter

from util import bin_dataset
from hypothesis_testing import ks2d
from broken_linear import broken_linear

__all__ = ["plot_binned", "chi2_bin"]

sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)


def plot_binned(
    data,
    sim=None,
    x_col="c",
    y_col="HOST_LOGMASS",
    fit=None,
    c_max_fit=2.0,
    bins=15,
    show_data=True,
    # split_mass=False,
    filename="",
    fig_options={},
    add_14J=False,
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
    c_max_fit : float
    bins : int
        Number of bins.
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
            # subsample down to size of data
            ax.plot(
                sim_x.sample(n=len(data_x), replace=False, random_state=4567),
                sim_y.sample(n=len(data_y), replace=False, random_state=4567),
                ".",
                color="tab:green",
                markersize=3,
                alpha=0.3,
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
            label="Binned " + fig_options.get("sim_name", "simulation"),
        )
    ax.errorbar(
        (data_edges[:-1] + data_edges[1:]) / 2,
        data_stat,
        yerr=data_error,
        color="tab:orange",
        fmt=">",
        label="Binned " + fig_options.get("data_name", "Pantheon+"),
    )

    if add_14J:
        ax.plot(1.163445, -17.843953, "b*", markersize=12, label="SN2014J")

    if fit is not None:
        _add_fit(ax, data_x, fit, c_max_fit)

    _add_fig_options(ax, x_col, y_col, fig_options)
    sns.despine()

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

    if x_col == "c":
        if y_col == "HOST_LOGMASS":
            y_coords = [14, 6, 6, 14]
        else:  # default to "x1_standardized"
            y_coords = [-24, -13, -13, -24]
        ax.fill(
            [-0.3, -0.3, 0.3, 0.3],
            y_coords,
            facecolor="black",
            edgecolor="none",
            alpha=0.1,
        )

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


def posterior_corner(posterior, var_names, filename=""):
    _, ax = new_figure()
    corner.corner(
        posterior[var_names],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        divergences=True,
        # truths={"x": 1.5, "y": [-0.3, 0.1]}
        smooth=1.25,
        labelpad=0.2,
        labels=[
            r"M$_0$",
            r"$\theta$",
            r"$\Delta_{\theta}$",
            r"$\sigma$",
            r"$\mu_c$",
            r"$\sigma_c$",
            r"$\alpha_c$",
        ],
    )
    savefig(
        "figures/" + filename, bbox_inches="tight"
    )  # removed bbox_inches="tight" from save_plot()


def plot_rms_c(data, fit, c_max_fit, bins, filename="rms-c.pdf"):
    """
    RMS of the residual to broken linear fit vs c.

    Parameters
    ----------
    data : pandas.DataFrame
    fit : pymc3 something, or is it the numpy
        `pymc3`.posterior.stack(draws=("chain", "draw"))
    c_max_fit : float
    bins : int
        Number of bins.
    filename : str
    """
    evenly_filled = True

    fig, ax = new_figure()

    ys = broken_linear(
        data["c"],
        fit["θ"].median().values,
        fit["Δ_θ"].median().values,
        fit["M_0"].median().values,
    )
    if evenly_filled:
        bins = 34  # 17 bins is ~100 objects per bin
        N = len(data["c"])
        bins = np.interp(np.linspace(0, N, bins + 1), np.arange(N), np.sort(data["c"]))

    data_stat, data_edges, binnumber = binned_statistic(
        data["c"],
        (data["x1_standardized"] - ys),
        statistic=lambda x: np.sqrt(np.mean(np.square(x))),
        # statistic=robust_scatter,
        bins=bins,
    )

    if evenly_filled:
        x_error = []
        for i in range(len(bins) - 1):
            x_error.append(
                (bins[i] + bins[i + 1]) / 2 - bins[i],
            )
        sp = ax.errorbar(
            (data_edges[:-1] + data_edges[1:]) / 2,
            data_stat,
            xerr=x_error,
            fmt=".",
        )
    else:
        counted = Counter(binnumber)
        N_per_bin = []
        for i in range(bins):
            N_per_bin.append(counted[i + 1])  # binnumber is 1 indexed.

        sp = ax.scatter(
            (data_edges[:-1] + data_edges[1:]) / 2,
            data_stat,
            marker=".",
            c=N_per_bin,
        )
        fig.colorbar(sp, label="Counts per Bin", orientation="vertical")
    ax.fill(
        [-0.3, -0.3, 0.3, 0.3],
        [max(data_stat) + 0.1, 0, 0, max(data_stat) + 0.1],
        facecolor="black",
        edgecolor="none",
        alpha=0.1,
    )
    ax.set_ylim(0, max(data_stat) + 0.06)
    ax.set_ylabel("RMS of Residuals (mag)")
    ax.set_xlabel("c")
    sns.despine()

    save_plot(filename)


def _add_fit_linmix(ax, fit, xs):
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


def _add_fit_broken_freq(ax, fit, xs):
    """
    fit : scipy.optimize.OptimizeResult
        Full output from something like scipy.optimize.minimize.
    """
    ys = broken_linear(pd.Series(xs), *fit.x)
    ax.plot(
        xs,
        ys,
        color="k",
        label=r"$\beta=$"
        + f"{np.tan(fit.x[0]):.2f}, "  # fit is over angle not slope
        + r"$\Delta \beta=$"
        + f"{np.tan(fit.x[1]):.2f}",
    )


def _add_fit_broken_bayes(ax, fit, xs):
    """
    fit : stacked arviz.InferenceData.posterior
        posterior part of the arviz.InferenceData. Should be
        `fit = InferenceData.posterior.stack(draws=("chain", "draw"))`
    """
    downsample_frac = 0.68  # 5--10% may be better for a long chain
    # // floors it but still keeps it as a float.
    for i in range(
        0,
        fit.sizes["draws"],
        round((fit.sizes["draws"] / downsample_frac) / fit.sizes["draws"]),
    ):
        ys = broken_linear(
            xs,
            fit["θ"][i].values,
            fit["Δ_θ"][i].values,
            fit["M_0"][i].values,
        )
        # if i == 0:
        #     # add one nearly blank line in the legned
        #     label = " "
        # else:
        #     label = None
        label = None
        ax.plot(xs, ys, color="0.5", alpha=0.005, label=label)
    ys = broken_linear(
        xs,
        fit["θ"].median().values,
        fit["Δ_θ"].median().values,
        fit["M_0"].median().values,
    )
    ax.plot(
        xs,
        ys,
        color="k",
        label=r"$\beta=$"
        + f"{np.tan(fit['θ'].median().values):.2f}"
        + r" $\pm$ "
        + f"{robust_scatter(np.tan(fit['θ'].values)):.2f}, "
        + r"$\Delta \beta=$"
        + f"{np.tan(fit['Δ_θ'].median().values):.2f}"
        + r" $\pm$ "
        + f"{robust_scatter(np.tan(fit['Δ_θ'].values)):.2f}",
    )


def _add_fit(ax, data_x, fit, c_max_fit):
    FIT_TYPE = "broken_bayes"  # TODO: Remove hard coding
    if c_max_fit > data_x.max():
        print(
            f"{c_max_fit = } is above {data_x.max() = }. Plot of linear fit will not go past the data.",
            end="\n\n",
        )
        c_max_fit = data_x.max()

    xs = np.arange(data_x.min(), c_max_fit, 0.01)

    if FIT_TYPE == "linmix":
        _add_fit_linmix(ax, fit, xs)
    elif FIT_TYPE == "broken_freq":
        _add_fit_broken_freq(ax, fit, xs)
    elif FIT_TYPE == "broken_bayes":
        _add_fit_broken_bayes(ax, fit, xs)


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
    print("Running KS2D ...")
    iterations = 500
    ks_p = np.empty(iterations)
    ks_p[0], n_samples = ks2d(data_x.values, data_y.values, sim_x.values, sim_y.values)
    for i in range(iterations - 1):
        ks_p[i + 1] = ks2d(
            data_x.values,
            data_y.values,
            sim_x.values,
            sim_y.values,
        )[0]

    print(f"For {filename},")
    print(
        f"2D KS-test: {np.median(ks_p):.3g} +/- {robust_scatter(ks_p):.3g}. With N=({n_samples})."
    )
    # replace all 1 item bins with nan.
    sim_stat[sim_error == 0.1] = np.nan
    data_stat[data_error == 0.1] = np.nan
    chi_numerator = (data_stat - sim_stat) ** 2
    chi_denominator = data_error**2
    chi_square = np.nansum(chi_numerator / chi_denominator)
    dof = np.count_nonzero(~np.isnan(chi_numerator / chi_denominator))
    print("Chi-2 numerator:", chi_numerator)
    print("Chi-2 denominator:", chi_denominator)
    print(f"Chi-2 binned stats: {chi_square:.3f},(N={dof}), {chi_square/dof:.3f}\n")
    # I want the Mann-Whitney-U of the data in each bin.
    # Then somehow compute a single value from the 25 Mann-Whitney-U values.


def bhm_diagnostic_plots(data, var_names=None):
    """arviz.InferenceData"""
    print(az.summary(data, var_names=var_names))

    az.plot_posterior(data, var_names=var_names)
    save_plot("diagnostics/posterior.pdf")

    az.plot_autocorr(
        data,
        var_names=var_names,
        combined=True,
    )
    save_plot("diagnostics/autocorr.pdf")

    az.plot_trace(data, var_names=var_names)
    save_plot("diagnostics/trace.png")

    ax = az.plot_ppc(data, observed=False, num_pp_samples=75, random_seed=42)
    sns.histplot(
        data.observed_data["c"].values, stat="density", ax=ax[0], label="Observed"
    )
    sns.histplot(data.observed_data["M'"].values, stat="density", ax=ax[1])
    ax[0].set_xlabel("c")
    ax[1].set_xlabel("M' (mag)")
    ax[0].legend(loc="upper right", fontsize="x-small")
    ax[1].legend().remove()
    sns.despine(left=True)
    save_plot("posterior_check.pdf")

    # az.plot_ppc(data, group="prior")
    # save_plot("prior_check.pdf")


def chi2_bin():
    bins = np.array([5, 7, 9, 11, 15, 20, 25, 50, 100])
    g10 = np.array([0.421, 0.268, 0.510, 0.461, 0.903, 1.742, 5.700, 0.690, 0.904])
    c11 = np.array([0.200, 0.085, 0.405, 0.291, 0.662, 0.372, 0.714, 0.904, 1.006])
    m11 = np.array([0.559, 0.382, 0.569, 0.462, 0.844, 0.842, 0.769, 0.642, 0.831])
    bs21 = np.array([0.176, 0.178, 0.450, 0.175, 0.263, 0.137, 0.230, 0.317, 0.429])
    # sim = np.aray([0.058, 11.731, 8.976, 8.508, 6.131, 2.134, 3.006, 95.944, 142.506])
    delta_c = (0.3 + 1.74) / bins

    _, ax = new_figure()

    ax.plot(bins, bs21, label="P22")
    ax.plot(bins, m11, ":", label="M11")
    ax.plot(bins, c11, ".-", label="C11")
    ax.plot(bins, g10, "--", label="G10")
    ax.set_xlabel("bins")
    ax.set_ylabel(r"$\chi^2_{\nu}$")
    ax.legend()
    sns.despine()

    save_plot("chi2_bins.pdf")

    _, ax = new_figure()

    ax.plot(delta_c, bs21, label="P22")
    ax.plot(delta_c, m11, ":", label="M11")
    ax.plot(delta_c, c11, "-.", label="C11")
    ax.plot(delta_c, g10, "--", label="G10")
    ax.invert_xaxis()
    ax.set_ylim(0.0, 2.1)
    ax.set_xlabel(r"$c$ bin width")
    ax.set_ylabel(r"$\chi^2_{\nu}$")
    ax.legend()
    sns.despine()

    save_plot("chi2_c.pdf")
