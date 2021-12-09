"""Module containing my figure creating functions
"""
import seaborn as sns
from matplotlib.pyplot import savefig
import numpy as np
import pandas as pd
import arviz as az
import corner

from br_util.plot import save_plot, new_figure
from br_util.stats import robust_scatter

from util import bin_dataset
from hypothesis_testing import ks2d
from broken_linear import broken_linear

__all__ = ["plot_binned"]

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
    bins = 15

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

    if fit is not None:
        _add_fit(ax, data_x, fit, c_max_fit)

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


def posterior_corner(posterior, var_names, filename=""):
    _, ax = new_figure()
    corner.corner(
        posterior[var_names],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        divergences=True,
        # truths={"x": 1.5, "y": [-0.3, 0.1]}
        smooth=0.75,
        labelpad=0.2,
        labels=[
            r"$M_0$",
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
    downsample_frac = 0.10  # 5--10% may be better for a long chain
    # // floors it but still keeps it as a float.
    for i in range(
        0,
        fit.sizes["draws"],
        round((fit.sizes["draws"] / downsample_frac) / fit.sizes["draws"]),
    ):
        ys = broken_linear(
            xs,
            fit["theta_cosmo"][i].values,
            fit["delta_theta"][i].values,
            fit["M0"][i].values,
        )
        if i == 0:
            # add one nearly blank line in the legned
            label = " "
        else:
            label = None
        ax.plot(xs, ys, color="0.5", alpha=0.02, label=label)
    ys = broken_linear(
        xs,
        fit["theta_cosmo"].median().values,
        fit["delta_theta"].median().values,
        fit["M0"].median().values,
    )
    ax.plot(
        xs,
        ys,
        color="k",
        label=r"$\beta=$"
        + f"{np.tan(fit['theta_cosmo'].median().values):.2f}"
        + r" $\pm$ "
        + f"{robust_scatter(np.tan(fit['theta_cosmo'].values)):.2f}, "
        + r"$\Delta \beta=$"
        + f"{np.tan(fit['delta_theta'].median().values):.2f}"
        + r" $\pm$ "
        + f"{robust_scatter(np.tan(fit['delta_theta'].values)):.2f}",
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
    ks_p, n_samples = ks2d(data_x.values, data_y.values, sim_x.values, sim_y.values)
    for _ in range(500):
        ks_p = np.append(
            ks_p,
            ks2d(
                data_x.values,
                data_y.values,
                sim_x.values,
                sim_y.values,
            )[0],
        )

    print(f"For {filename},")
    print(
        f"2D KS-test: {np.median(ks_p):.3g} +/- {robust_scatter(ks_p):.3g}. With N=({n_samples})."
    )
    chi_square = np.nansum(
        (data_stat - sim_stat) ** 2 / (sim_error ** 2 + data_error ** 2)
    )
    print(f"Chi-2 binned stats: {chi_square:.3f}\n")
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

    ax = az.plot_ppc(data, observed=False, num_pp_samples=45, random_seed=42)
    sns.histplot(
        data.observed_data["c"].values, stat="density", ax=ax[0], label="Observed"
    )
    sns.histplot(data.observed_data["M'"].values, stat="density", ax=ax[1])
    ax[0].set_xlabel("c")
    ax[1].set_xlabel("M' (mag)")
    ax[0].legend(loc="upper right")
    save_plot("posterior_check.pdf")

    # az.plot_ppc(data, group="prior")
    # save_plot("prior_check.pdf")
