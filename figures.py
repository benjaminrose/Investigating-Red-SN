"""Module containing my figure creating functions
"""
from collections import Counter
from pickle import TRUE
from pkgutil import extend_path

import seaborn as sns
from matplotlib.pyplot import savefig
import numpy as np
import pandas as pd
import arviz as az
import corner
from scipy.stats import binned_statistic
from statsmodels.nonparametric.kernel_regression import KernelReg

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

    # if sim is not None:
    #     _data_v_sim(
    #         data_x,
    #         data_y,
    #         data_stat,
    #         data_error,
    #         sim_x,
    #         sim_y,
    #         sim_stat,
    #         sim_error,
    #         bins,
    #         filename,
    #         fig_options,
    #     )


def posterior_corner(posterior, var_names, filename=""):
    _, ax = new_figure()
    corner.corner(
        posterior[var_names],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        divergences=True,
        title_fmt=".3f",
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
        p = np.polyfit(
            (data_edges[:-1] + data_edges[1:]) / 2,
            data_stat,
            2,
        )
        print(f"Polly fit for RMS-c: {p}")
        print(
            "Function for chi^2 hard codes this as: -2.60c**3 + 2.81c**2 + 0.31c + 0.17"
        )

        # I want to define this in one function and use it in another.
        global kr
        kr = KernelReg(data_stat, (data_edges[:-1] + data_edges[1:]) / 2, "c")

        xs = np.linspace(
            (data_edges[0] + data_edges[1]) / 2,
            (data_edges[-2] + data_edges[-1]) / 2,
            100,
        )
        ax.plot(xs, kr.fit(xs)[0], label="Polynomial Fit")

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
    data,
    sim_stat,
    # sim_error,
    Nbins,
    reduced=True,
):
    """
    data_, sim_: pandas.Series

    Return
    ------
    float:
        Reduced Chi^2 value between `data` and `sim`.
    """

    # NEW chi-squared method.
    # add a column that states the c-bin number. Use that number to pull from the binned sim_stat
    data["c_bins"] = pd.cut(data["c"], Nbins, labels=range(Nbins))
    theory_y_values = []
    theory_error = []
    for i in data["c_bins"].values:
        theory_y_values.append(sim_stat[i])
        # theory_error.append(sim_error[i])
    theory_y_values = np.array(theory_y_values)
    # theory_error = np.array(theory_error)

    chi_numerator = (data["x1_standardized"].values - theory_y_values) ** 2
    # chi_denominator = _rms(data["c"]) ** 2
    chi_denominator = kr.fit(data["c"])[0] ** 2
    if False:
        new_figure()
        ax = sns.histplot(
            data=chi_numerator / chi_denominator,
            color="b",
            cumulative=False,
            bins="auto",
            element="step",
            fill=False,
        )
        sns.rugplot(
            data=chi_numerator / chi_denominator,
            height=-0.02,
            lw=1,
            clip_on=False,
            ax=ax,
        )
        ax.set_xlabel("chi^2")
        print(len(chi_numerator))
        # print(
        #     data.loc[
        #         chi_numerator / chi_denominator > 5,
        #         ["zHD", "c", "x1_standardized", "cERR", "x1_standardized_ERR"],
        #     ]
        # )
        # print((chi_numerator / chi_denominator)[chi_numerator / chi_denominator > 5])
        # print(
        #     chi_numerator[chi_numerator / chi_denominator > 5],
        #     chi_denominator[chi_numerator / chi_denominator > 5],
        # )
        # print(
        #     "0.5 < c < 1.0\n",
        #     data.loc[
        #         np.logical_and(0.5 < data["c"], data["c"] < 1.0),
        #         ["zHD", "c"],
        #     ],
        #     "\n",
        #     chi_numerator[np.logical_and(0.5 < data["c"], data["c"] < 1.0)],
        #     "\n",
        #     chi_denominator[np.logical_and(0.5 < data["c"], data["c"] < 1.0)],
        #     "\n",
        #     data.iloc[
        #         [np.logical_and(0.5 < data["c"], data["c"] < 1.0)], "x1_standardized"
        #     ].values,
        #     "\n",
        #     theory_y_values[np.logical_and(0.5 < data["c"], data["c"] < 1.0)],
        # )

        from matplotlib.pyplot import show

        show()
        import sys

        sys.exit()
    # a nan in sim_stat will propigate to the chi_square.
    chi_square = np.nansum(chi_numerator / chi_denominator)
    # use non-nan length of theory_y_values over len(data) to account for dropped data that is in an empty sim bin
    if reduced:
        dof = np.count_nonzero(~np.isnan(theory_y_values))
    else:
        dof = 1.0

    # print(f"Chi-2 numerator {chi_numerator}")
    # print(f"Chi-2 denominator {chi_denominator.values}")
    # print(f"Chi-2 binned stats: {chi_square:.3f},(N={dof}), {chi_square/dof:.3f}\n")
    return chi_square / dof


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


def _rms(c, degree=2):
    # 3-degrees, -2.60842542 * c**3 + 2.81549634 * c**2 + 0.31497116 * c + 0.1741919
    # 2-degrees, 0.30437141 * c**2 + 0.25717747 * c + 0.19878428
    if degree == 3:
        return -2.60842542 * c ** 3 + 2.81549634 * c ** 2 + 0.31497116 * c + 0.1741919
    else:
        return 0.30437141 * c ** 2 + 0.25717747 * c + 0.19878428


def chi2_bin(data, g10, c11, m11, bs21):
    x_col, y_col = "c", "x1_standardized"
    # bins = np.array([9, 11, 15, 20, 25, 50, 100])
    bins = np.array([20, 25, 50, 100])

    g10_chi = []
    c11_chi = []
    m11_chi = []
    bs21_chi = []
    iteration = 0
    for sim in [g10, c11, m11, bs21]:
        for bin in bins:
            print(iteration, bin)
            # define bins on data, not on sims.
            _, _, _, data_edges, _ = bin_dataset(data, x_col, y_col, bins=bin)
            # get median values of the sims, pass this to _data_v_sim.
            _, _, sim_stat, _, _ = bin_dataset(sim, x_col, y_col, bins=data_edges)
            # Keep uncertainties fixed and not floating between data sets.
            # _, _, _, _, sim_error = bin_dataset(bs21, x_col, y_col, bins=data_edges)

            reduced_chi = _data_v_sim(
                data,
                sim_stat,
                # sim_error,
                Nbins=bin,
            )
            if iteration == 0:
                g10_chi.append(reduced_chi)
            elif iteration == 1:
                c11_chi.append(reduced_chi)
            elif iteration == 2:
                m11_chi.append(reduced_chi)
            elif iteration == 3:
                bs21_chi.append(reduced_chi)
            else:
                raise RuntimeError("IDK, something failed.")
        iteration += 1
    # print as a vertical table so it can be added to the latex table.
    print("## Reduced chi-square")
    print("g10")
    for i in g10_chi:
        print(format(i, ".3f"))
    print("c11")
    for i in c11_chi:
        print(format(i, ".3f"))
    print("m11")
    for i in m11_chi:
        print(format(i, ".3f"))
    print("bs21")
    for i in bs21_chi:
        print(format(i, ".3f"))

    bs21_m11_chi = []
    for bin in bins:
        # define bins on data, not on sims.
        _, _, _, data_edges, _ = bin_dataset(data, x_col, y_col, bins=bin)
        # get median values of the sims, pass this to _data_v_sim.
        _, _, sim_stat, _, _ = bin_dataset(sim, x_col, y_col, bins=data_edges)
        # Keep uncertainties fixed and not floating between data sets.
        # _, _, _, _, sim_error = bin_dataset(bs21, x_col, y_col, bins=data_edges)

        reduced_chi = _data_v_sim(
            data,
            sim_stat,
            # sim_error,
            Nbins=bin,
        )
        bs21_m11_chi.append(reduced_chi)
    print("bs21 vs m11")
    for i in bs21_m11_chi:
        print(format(i, ".3f"))

    delta_c = (data["c"].max() - data["c"].min()) / bins

    # _, ax = new_figure()
    # ax.plot(bins, bs21, label="P22")
    # ax.plot(bins, m11, ":", label="M11")
    # ax.plot(bins, c11, ".-", label="C11")
    # ax.plot(bins, g10, "--", label="G10")
    # ax.set_xlabel("bins")
    # ax.set_ylabel(r"$\chi^2_{\nu}$")
    # ax.legend()
    # sns.despine()
    # save_plot("chi2_bins.pdf")

    _, ax = new_figure()

    ax.plot(delta_c, bs21_chi, label="P22")
    ax.plot(delta_c, m11_chi, ":", label="M11")
    ax.plot(delta_c, c11_chi, "-.", label="C11")
    ax.plot(delta_c, g10_chi, "--", label="G10")
    # ax.set_title(f"c > {data['c'].min():.2f}")
    ax.invert_xaxis()
    # ax.set_ylim(0.0, 2.1)
    ax.set_xlabel(r"$c$ bin width")
    ax.set_ylabel(r"$\chi^2_{\nu}$")
    ax.legend()
    sns.despine()

    save_plot("chi2_c_width.pdf")


def chi_color_min(data, g10, c11, m11, bs21, min=True):
    x_col, y_col = "c", "x1_standardized"
    bin_spacing = 0.1  # matchs 20 bins over the full c-range of the data
    # color_mins = np.array([-0.3, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.7])
    color_mins = np.array([-0.3, 0.3, 1.7])
    # color_mins = np.array([0.7, 1.7])
    min = False

    g10_chi = []
    c11_chi = []
    m11_chi = []
    bs21_chi = []
    iteration = 0
    for sim in [g10, c11, m11, bs21]:
        # for sim in [c11]:
        for i, c_min in enumerate(color_mins[:-1]):
            if min:
                c_max = data["c"].max()
            else:
                c_max = color_mins[i + 1]
            bin = round((c_max - c_min) / bin_spacing)
            print(f"{iteration}: {c_min=} {c_max=} {bin=}")
            # define bins on data, not on sims.
            _, _, _, data_edges, _ = bin_dataset(
                data[np.logical_and(c_min < data[x_col], data[x_col] < c_max)],
                x_col,
                y_col,
                bins=bin,
            )
            # get median values of the sims, pass this to _data_v_sim.
            _, _, sim_stat, _, _ = bin_dataset(
                sim[np.logical_and(c_min < sim[x_col], sim[x_col] < c_max)],
                x_col,
                y_col,
                bins=data_edges,
            )

            reduced_chi = _data_v_sim(
                data[np.logical_and(c_min < data[x_col], data[x_col] < c_max)],
                sim_stat,
                Nbins=bin,
                # reduced=False,
            )
            if iteration == 0:
                g10_chi.append(reduced_chi)
            elif iteration == 1:
                c11_chi.append(reduced_chi)
            elif iteration == 2:
                m11_chi.append(reduced_chi)
            elif iteration == 3:
                bs21_chi.append(reduced_chi)
            else:
                raise RuntimeError("IDK, something failed.")
        iteration += 1
    # print as a vertical table so it can be added to the latex table.
    print("## Reduced chi-square")
    print("color min | g10")
    for i, chi in enumerate(g10_chi):
        print(color_mins[i], " | ", format(chi, ".3f"))
    print("color min | c11")
    for i, chi in enumerate(c11_chi):
        print(color_mins[i], " | ", format(chi, ".3f"))
    print("color min | m11")
    for i, chi in enumerate(m11_chi):
        print(color_mins[i], " | ", format(chi, ".3f"))
    print("color min | bs21")
    for i, chi in enumerate(bs21_chi):
        print(color_mins[i], " | ", format(chi, ".3f"))

    _, ax = new_figure()

    if min:
        ax.plot(color_mins[:-1], bs21_chi, label="P22")
        ax.plot(color_mins[:-1], m11_chi, ":", label="M11")
        ax.plot(color_mins[:-1], c11_chi, "-.", label="C11")
        ax.plot(color_mins[:-1], g10_chi, "--", label="G10")
        ax.set_xlabel(r"$c$ Minimum")
    else:
        x_error = []
        for i in range(len(color_mins) - 1):
            x_error.append(
                (color_mins[i] + color_mins[i + 1]) / 2 - color_mins[i],
            )
        ax.errorbar(
            (color_mins[:-1] + color_mins[1:]) / 2,
            c11_chi,
            xerr=x_error,
            fmt="^",
            label="C11",
        )
        ax.errorbar(
            (color_mins[:-1] + color_mins[1:]) / 2,
            g10_chi,
            xerr=x_error,
            fmt="o",
            label="G10",
        )
        ax.errorbar(
            (color_mins[:-1] + color_mins[1:]) / 2,
            m11_chi,
            xerr=x_error,
            fmt="x",
            label="M11",
        )
        ax.errorbar(
            (color_mins[:-1] + color_mins[1:]) / 2,
            bs21_chi,
            xerr=x_error,
            fmt=".",
            label="P22",
        )
        ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$\chi^2_{\nu}$")
    ax.legend(loc="lower right")
    sns.despine()

    if min:
        save_plot("chi2_c_min.pdf")
    else:
        save_plot("chi2_c_binned.pdf")
