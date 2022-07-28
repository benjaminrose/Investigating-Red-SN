"""Utility functions"""

__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter

from scipy.stats import binned_statistic
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import matplotlib.pyplot as plt

from br_util.plot import save_plot, new_figure
from br_util.stats import robust_scatter

from hypothesis_testing import ks1d
from broken_linear import broken_linear


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
    data = data.dropna(subset=[x, y])
    data_x_axis = data[x]
    data_y_axis = data[y]

    data_stat, data_edges, binnumber = binned_statistic(
        data_x_axis, data_y_axis, statistic="median", bins=bins
    )
    data_error, _, _ = binned_statistic(
        data_x_axis, data_y_axis, statistic=error_stat, bins=bins
    )
    # data_error[data_error == 0] = SCATTER_FLOOR
    for i, binned_error_bar in enumerate(data_error):
        # if no error but there is data, then there is only one datapoint in bin.
        if binned_error_bar == 0 and not np.isnan(data_stat[i]):
            in_bin = np.logical_and(
                # From https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
                # All but the last (righthand-most) bin is half-open. In other words,
                # if bins is [1, 2, 3, 4], then the first bin is [1, 2)
                # (including 1, but excluding 2) and the second [2, 3). The last bin,
                # however, is [3, 4], which includes 4.
                # From me, somehow, it fails unless I invert it (1, 2].
                data_edges[i] < data_x_axis,
                data_x_axis <= data_edges[i + 1],
            )
            data_error[i] = data.loc[
                in_bin, y + "_ERR"
            ].values  # assume this is "x1_standardized_ERR"
    return (
        data_x_axis,
        data_y_axis,
        data_stat,
        data_edges,
        data_error,
        Counter(binnumber),
    )


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
        "--break_value",
        type=float,
        default=0.3,
        help="c-value to apply broken linear fit (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--cmax",
        type=float,
        default=1.3244,  # max in BS21 on Oct 22, 2021
        help="maximum c used in fitting BETA (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--zmin",
        type=float,
        default=0.0,
        help="Minimum redshift to use (default: %(default)s)",
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
        "--all",
        action="store_true",
        default=False,
        help="include parts that are no longer in main analysis (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--talk",
        action="store_true",
        default=False,
        help="make simpiler figures (default: %(default)s)",
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


def calc_chi2(data, g10, c11, m11, bs21, kr_data, kr_sims):
    """
    ...
    kr_data : Kernel Regression
        Regression of the RMS of the data. Originally from `figures.plot_rms_c()`.
    kr_sims : dict
        Dictionary (keys of ["g10", "c11", "m11", "bs21"]) Kernel Regressions of
        the predicted RMS from the four scatter models. Originally from `kr_sims()`.
    """
    data_c = data["c"].values
    data_m_prime = data["x1_standardized"].values
    bin_edges = np.linspace(-0.3, max(data_c), 21)
    print(f"{len(data_m_prime)=}")
    for model_name, sim in zip(
        ["g10", "c11", "m11", "bs21"],
        [g10, c11, m11, bs21],
    ):
        sim_c = sim["c"].values
        sim_m_prime = sim["x1_standardized"].values
        # equations 6 and 8 from P22
        chi_c = _chi_c(data_c, sim_c, bin_edges)
        chi_mprime = _chi_mprime(
            data_c, data_m_prime, kr_data, kr_sims[model_name], model_name
        )
        chi = chi_c + chi_mprime
        print(f"{model_name}, {chi_c:.2f}, {chi_mprime:.2f}")


def _chi_c(data, sim, bins):
    # equation 6 from P22
    N_data, _ = np.histogram(data, bins)
    N_sim, _ = np.histogram(sim, bins)
    # scale sim to match data
    N_sim = N_sim * np.sum(N_data) / np.sum(N_sim)
    e_poisson = np.sqrt(N_data)
    # if no data, set error to 1, still test that sim matches the no data.
    e_poisson[N_data == 0] = 1
    # e_poisson[~np.isfinite(e_poisson)] = 1 # Idk why Brodie had this. I am skipping it for now.

    if debug := False:
        # indexes for `sim` to re-sample `sim` to be the length of `data`.
        subsample = np.random.randint(0, len(sim), len(data))
        print(f"KS test {ks1d(data, sim[subsample]):.4f}")

    return np.sum((N_data - N_sim) ** 2 / e_poisson**2)


def _chi_mprime(data_c, data_m_prime, kr_data, kr_sim, sim_name=None):

    # OLD per bin chi^2
    # (data_c, data_m_prime, sim_c, sim_m_prime, bins):
    # # equation 8 from P22
    # m_prime_data, _, _ = binned_statistic(
    #     data_c, data_m_prime, statistic="median", bins=bins
    # )
    # m_prime_scatter, _, _ = binned_statistic(
    #     data_c, data_m_prime, statistic=robust_scatter, bins=bins
    # )
    # # QUESTION: should I set nan values here to be same as bad errors, or 10x bad errors?
    # m_prime_sim, _, _ = binned_statistic(
    #     sim_c, sim_m_prime, statistic="median", bins=bins
    # )
    # # QUESTION: This number is very important. Should I do the fitting and just let the first chi square handle this issue?
    # m_prime_sim[np.isnan(m_prime_sim)] = -15.5
    # N_data, _ = np.histogram(data_c, bins)
    # # QUESTION: what should we do when N_data is zero?
    # # Does not matter, since m_prime_data will be nan for these values.
    # N_data[N_data == 0] = 1
    # # QUESTION: why is it robust scatter and not robust scatter squared?
    # e_m_prime_sqr = m_prime_scatter / np.sqrt(N_data)
    # # if 0 or 1 datum in a bin, set error to 1, still test that sim matches the no data.
    # # QUESTION: Should this be 1, or the global RMS value?
    # e_m_prime_sqr[m_prime_scatter == 0] = 0.2

    # equation 8 from P22
    m_prime_sim = kr_sim.fit(data_c)[0]

    # N_data, _ = np.histogram(data_c, bins)
    # # QUESTION: what should we do when N_data is zero?
    # # Does not matter, since m_prime_data will be nan for these values.
    # N_data[N_data == 0] = 1
    # # QUESTION: why is it robust scatter and not robust scatter squared?
    # e_m_prime_sqr = m_prime_scatter / np.sqrt(N_data)
    # # if 0 or 1 datum in a bin, set error to 1, still test that sim matches the no data.
    # # QUESTION: Should this be 1, or the global RMS value?
    # e_m_prime_sqr[m_prime_scatter == 0] = 0.2
    e_m_prime_sqr = kr_data.fit(data_c)[0]

    # nansum to ignore nan's in m_prime_data
    chi = np.nansum(
        (data_m_prime - m_prime_sim - np.median(data_m_prime - m_prime_sim)) ** 2
        / e_m_prime_sqr**2
    )
    if debug := False:
        _, ax = new_figure()
        ax.hist(data_m_prime - m_prime_sim - np.median(data_m_prime - m_prime_sim))
        ax.set_yscale("log")
        save_plot(f"chi_m_{max(data_c):.2f}_{sim_name}.pdf")
    # print("offsets:", np.median(data_m_prime - m_prime_sim), "mag")
    return chi


def kr_sims(g10, c11, m11, bs21, c_break, var_name="x1_standardized", fit=None):
    """
    use as kr["g10"].bins(data["c"])[0] or
    xs = np.linespace(min, max, 100)
    kr["bs21"].fit(xs)[0]
    """
    kr = {}
    bin_edges = np.linspace(-0.3, 1.7, 34 + 1)  # edges is n_bins + 1
    for model_name, simulation in zip(
        ["g10", "c11", "m11", "bs21"],
        [g10, c11, m11, bs21],
    ):
        if fit is not None:
            ys = broken_linear(
                simulation["c"],
                fit["θ"].median().values,
                fit["Δ_θ"].median().values,
                fit["M'_0"].median().values,
                c_break=c_break,
            )
            y = simulation["x1_standardized"] - ys
            stat = lambda x: np.sqrt(np.mean(np.square(x)))
        else:
            y = simulation["x1_standardized"]
            stat = "median"
        stat, edges, _ = binned_statistic(
            simulation["c"],
            (y),
            statistic=stat,
            bins=bin_edges,
        )

        # KernelReg can not take nan values, or the whole KR is nan.
        kr[model_name] = KernelReg(
            stat[~np.isnan(stat)],
            ((edges[:-1] + edges[1:]) / 2)[~np.isnan(stat)],
            "c",
        )

    return kr
