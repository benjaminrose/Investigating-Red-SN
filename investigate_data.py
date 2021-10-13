""" investigate_red_sn.py

Requires python 3.9
"""
__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from astropy.cosmology import wCDM
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, ks_2samp
import seaborn as sns

import linmix

from br_util.stats import robust_scatter
from br_util.snana import read_data
from br_util.plot import save_plot, new_figure

sns.set(context="talk", style="ticks", font="serif", color_codes=True)


class Fitres:
    # should this subclass pandas?
    def __init__(self, file_path, alpha=0.15, verbose=False):
        self.data = read_data(file_path, 1, verbose)
        self.VERBOSE = verbose
        # Keywords used in plotting both data sets
        # So not: data, labels, colors, ...
        self.histplot_keywords = {
            "bins": "auto",
            "cumulative": True,
            "element": "step",
            "fill": False,
            "stat": "density",
        }
        self.M0 = -19.34
        self.alpha = alpha
        self.beta = 3.1

    def calc_HR(self):
        """Adds columns "mu_theory", "x1_standardized", & "HR_naive"

        Parameters
        ----------
        data: pandas.DataFrame
            `data` needs columns "mB", "x1", "c". Data is assumed to be a
            DataFrame of an SNANA fitres file.
        M0:
        alpha:
        beta:
        """
        self.data["mu_theory"] = COSMO.distmod(self.data["zHD"]).value
        self.data["x1_standardized"] = (
            self.data["mB"] - self.data["mu_theory"] - self.alpha * self.data["x1"]
        )
        self.data["x1_standardized_ERR"] = np.sqrt(
            # mb error, no cosmology error, x1 error
            self.data["mBERR"] ** 2
            + 0
            + (self.alpha * self.data["x1ERR"]) ** 2
        )
        self.data["HR_naive"] = (
            self.data["mB"]
            + self.alpha * self.data["x1"]
            - self.beta * self.data["c"]
            + self.M0
            - self.data["mu_theory"]
        )

        if self.VERBOSE:
            print(
                f"Naive distances:\n{self.data[['zHD', 'mu_theory', 'x1_standardized', 'HR_naive']].head(10)}\n"
            )
        return self.data

    def calc_RV(self):
        self.data["RV"] = (self.data["x1_standardized"] + self.M0) / (
            self.data["c"] + 0.1
        )
        return self.data

    def clean_data(self, x1err_max=2.0, x1_max=5, cerr_max=1, c_min=-0.5):
        self._cut_x1err(x1err_max)
        self._cut_x1(x1_max)
        self._cut_cerr(cerr_max)
        self._cut_c(c_min)
        return self

    def _cut_c(self, c_min):
        if self.VERBOSE:
            print(f"Initial c distribution:\n{self.data[['c', 'cERR']].describe()}\n")
        self.data = self.data[c_min < self.data["c"]]
        if self.VERBOSE:
            print(
                f"After-cuts c distribution:\n{self.data[['c', 'cERR']].describe()}\n"
            )
        return self.data

    def _cut_cerr(self, cerr_max):
        # TODO: refactor to pass value and key rather than a function for both x1 and c.
        if self.VERBOSE:
            print(
                f"Initial cERR distribution:\n{self.data[['c', 'cERR']].describe()}\n"
            )
        self.data = self.data[np.abs(self.data["cERR"]) <= cerr_max]
        if self.VERBOSE:
            print(
                f"After-cuts cERR distribution:\n{self.data[['c', 'cERR']].describe()}\n"
            )
        return self.data

    def _cut_x1(self, x1_max):
        if self.VERBOSE:
            print(f"Initial x1 distribution:\n{self.data['x1'].describe()}\n")
        self.data = self.data[np.abs(self.data["x1"]) <= x1_max]
        if self.VERBOSE:
            print(f"After-cuts x1 distribution:\n{self.data['x1'].describe()}\n")
        return self.data

    def _cut_x1err(self, x1err_max):
        if self.VERBOSE:
            print(
                f"Initial cERR distribution:\n{self.data[['x1', 'x1ERR']].describe()}\n"
            )
        self.data = self.data[np.abs(self.data["x1ERR"]) <= x1err_max]
        if self.VERBOSE:
            print(
                f"After-cuts cERR distribution:\n{self.data[['x1', 'x1ERR']].describe()}\n"
            )
        return self.data

    def plot_beta_ben(self, save=True):
        fig, ax = new_figure()

        # y = data["mB"] - data["mu_theory"] - alpha * data["x1"]
        x_main = np.linspace(self.data["c"].min(), self.data["c"].max(), 50)
        x_alt1 = np.linspace(0.15, 0.8, 50)
        x_alt2 = np.linspace(0.8, self.data["c"].max(), 20)
        beta1 = 3.5
        M0_1 = 0.25
        beta2 = 1.0
        M0_2 = 2.0

        ax.plot(self.data["c"], self.data["x1_standardized"], ".", alpha=0.7)
        ax.plot(
            x_main,
            self.beta * x_main + self.M0,
            label=r"$\beta$" + f"={self.beta}, M0={self.M0}",
        )
        ax.plot(
            x_alt1,
            beta1 * x_alt1 + self.M0 + M0_1,
            label=r"$\beta$" + f"={beta1}, M0={self.M0+M0_1}",
        )
        ax.plot(
            x_alt2,
            beta2 * x_alt2 + self.M0 + M0_2,
            label=r"$\beta$" + f"={beta2}, M0={self.M0+M0_2}",
        )

        ax.set(
            xlabel="Apparent Color c",
            ylabel=f"$ m_B - \mu(z) + ${self.alpha} $x_1$ (mag)",
        )
        ax.invert_yaxis()
        plt.legend(fontsize="smaller")

        save_plot("color-luminosity-naive.pdf")

    def plot_fitprob_c(self, fitprob_cut=0.01):
        _, ax = new_figure()

        fail = self.data["FITPROB"] < fitprob_cut
        ax.plot(self.data["c"], self.data["FITPROB"], ".", label="Passes")
        ax.plot(
            self.data.loc[fail, "c"], self.data.loc[fail, "FITPROB"], ".", label="Fails"
        )

        ax.set_xlabel("SALT2 c")
        ax.set_ylabel("SNANA Fit Probability")
        plt.legend()
        save_plot("color-fitprob.pdf")

    def plot_fitprob_binned_c(self, fitprob_cut=0.01):
        # data, x, y, c_min=-0.5, bins=25, error_stat=robust_scatter
        # data_x, data_y, data_stat, data_edges, data_error = bin_dataset(self.data, "c",

        data, x, y, bins = self.data, "c", "FITPROB", 25  ## passed to bin_dataset
        data_mask = data[x] > -0.5
        data_x_axis = data.loc[data_mask, x]
        data_y_axis = data.loc[data_mask, y]
        statistic = lambda x: len(x[x > fitprob_cut]) / len(
            x
        )  # Not possible with bin_dataset

        data_stat, data_edges, _ = binned_statistic(
            data_x_axis, data_y_axis, statistic=statistic, bins=bins
        )
        # data_error, _, _ = binned_statistic(
        #     data_x_axis, data_y_axis, statistic=error_stat, bins=bins
        # )

        _, ax = new_figure()

        ax.errorbar(
            (data_edges[:-1] + data_edges[1:]) / 2,
            data_stat * 100,  # Make a percent
            fmt="*",
            label=f"Percent with FITPROB > {fitprob_cut}",
        )

        ax.set_xlabel("c")
        ax.set_ylabel("Percent pass")
        plt.legend()
        save_plot("fitprob_c.pdf")

    def plot_hist(self, key, filename=""):
        new_figure()
        ax = sns.histplot(
            data=self.data[~self.data.index.duplicated(keep="first")],
            x=key,
            color="b",
            cumulative=False,
            bins="auto",
            element="step",
            # **self.histplot_keywords,
        )
        if key == "c":
            ax.set_yscale("log")
        save_plot(filename)

    def plot_hist_c_special(self, key, filename=""):
        new_figure()
        ax = sns.histplot(
            data=np.log(self.data[key] - self.data[key].min() - 0.0001),
            # x=key,
            color="b",
            cumulative=False,
            # **self.histplot_keywords,
        )
        ax.set_xlabel(f"log(c +{self.data[key].min() +0.0001:+.3f})")
        save_plot(filename)

    def plot_hists(self, key="FITPROB", filename=""):
        new_figure()

        ax = sns.histplot(
            data=self.blue_subsample[
                ~self.blue_subsample.index.duplicated(keep="first")
            ],
            label=f"c <= {self.c_split}",
            x=key,
            color="b",
            **self.histplot_keywords,
        )
        sns.histplot(
            data=self.red_subsample[~self.red_subsample.index.duplicated(keep="first")],
            x=key,
            ax=ax,
            label=f"c > {self.c_split}",
            color="r",
            **self.histplot_keywords,
        )
        plt.legend()
        save_plot(filename)

    def slipt_on_c(self, c_split):
        """
        Parameters
        ----------
        c_split: float
            Value in c where <= is classified as blue and > is classified as red.

        Returns
        -------
        blue_subsample: pd.DataFrame
            A subsample of self.data where c <= `c_split`.
        red_subsample: pd.DataFrame
            A subsample of self.data where `c_split` < c.
        """
        self.c_split = c_split
        self.blue_subsample = self.data[self.data["c"] <= self.c_split].copy()
        self.red_subsample = self.data[self.data["c"] > self.c_split].copy()

    def to_file(self, filename, df=None):
        """
        Save `Fitres` object into an SNANA compatible ascii file.

        Parameters
        ----------
        filename: str, pathlib.Path
            ...
        df: pandas.DataFrame
            Can pass a dataframe that you would like to be saved into
        """

        if df is None:
            df = self.data

        # add SNANA keys/index in the correct order for each row.
        df.reset_index(inplace=True)
        varnames = ["SN:"] * df.shape[0]
        df.insert(loc=0, column="VARNAMES:", value=varnames)

        df.to_csv(filename, sep=" ", index=False)


def bin_dataset(data, x, y, x_min=-0.5, bins=25, error_stat=robust_scatter):
    """
    Parameters
    -----------
    data: pandas.DataFrame
    x: str
        column for x-axis (binning-axis)
    y: str
        column name for y-axis (statistic axis)
    x_min: float
        minimum value of x column, used in data masking. Defaults to -0.5.
    bins: int or sequence of scalars
        passed to scipy.stats.binned_statistic. Defaults to 25 bins
        (Not the same as scipy's default.)
    stat: str, function
        Passed to scipy.stats.binned_statistic. Defaults to
        `br_util.stats.robust_scatter`.
    """
    data_mask = data[x] > x_min
    data_x_axis = data.loc[data_mask, x]
    data_y_axis = data.loc[data_mask, y]

    data_stat, data_edges, _ = binned_statistic(
        data_x_axis, data_y_axis, statistic="median", bins=bins
    )
    data_error, _, _ = binned_statistic(
        data_x_axis, data_y_axis, statistic=error_stat, bins=bins
    )
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
                markersize=3,
                alpha=0.3,
                label=fig_options.get("sim_name", "Simulation"),
            )
        ax.plot(
            data_x,
            data_y,
            ".",
            markersize=3,
            alpha=0.3,
            label=fig_options.get("data_name", "Pantheon+"),
        )
    if sim is not None:
        ax.errorbar(
            (sim_edges[:-1] + sim_edges[1:]) / 2,
            sim_stat,
            yerr=sim_error,
            fmt="^",
            label="Binned simulation",
        )
    ax.errorbar(
        (data_edges[:-1] + data_edges[1:]) / 2,
        data_stat,
        yerr=data_error,
        fmt=">",
        label="Binned data",
    )

    if fit is not None:
        xs = np.arange(data_x.min(), data_x.max())
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

    if fig_options.get("y_flip", False):
        ax.invert_yaxis()
    ax.set_xlabel(fig_options.get("x_label", x_col))
    ax.set_ylabel(fig_options.get("y_label", y_col))
    ax.set_ylim(fig_options.get("ylim", ax.get_ylim()))
    ax.legend(fontsize="x-small", loc=fig_options.get("leg_loc", "best"))

    save_plot(filename)

    ks, p = ks_2samp(data_stat, sim_stat)
    print(f"For {filename},\nKS-test of binned stats: {ks}, p={p:.5f}")
    chi_square = np.nansum(
        (data_stat - sim_stat) ** 2 / (sim_error ** 2 + data_error ** 2)
    )
    print(f"reduced chi-2 binned stats: {chi_square/bins:.3f}\n")
    # I want the Mann-Whitney-U of the data in each bin.
    # Then somehow compute a single value from the 25 Mann-Whitney-U values.


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose output"
    )
    arg_parser.add_argument("--version", action="version", version=__version__)
    arg_parser.add_argument(
        "--rv", action=BooleanOptionalAction, default=False
    )  # python 3.9
    arg_parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="used for light-curve shape standardization (default: %(default)s)",
    )
    cli = arg_parser.parse_args()

    # Inputs
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = cli.verbose
    RUN_LINMIX = cli.rv
    LINMIX_MIN_ITR = (
        2000  # defaults to 5000, 2000 is good for our our data set, 500 for fast
    )
    print(cli.rv)
    c_splits = [0.3]  # was [0.1, 0.3] during initial analysis
    x1err_max = 1.0
    x1_max = 3  # x1 cut is on abs(x1)
    cerr_max = 0.2
    c_min = -0.3
    # fitprob_min = 0.1
    alpha = cli.alpha
    COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)

    # Import and clean data
    ####
    data = Fitres(data_file, alpha, VERBOSE)
    data.clean_data(x1err_max, x1_max, cerr_max, c_min)
    data.calc_HR()
    data.calc_RV()

    print(f"There are {data.data.loc[data.data['c']>0.3, 'c'].shape[0]} SN with c>0.3")

    data.slipt_on_c(0.99)
    print("SN affected by c=1 boundry.")
    print(
        data.red_subsample[
            [
                "CIDint",
                "IDSURVEY",
                "c",
                "cERR",
                "x1_standardized",
                "x1_standardized_ERR",
                "RV",
            ]
        ]
    )

    if RUN_LINMIX:
        lm = linmix.LinMix(
            x=data.data["c"],
            y=data.data["x1_standardized"],
            xsig=data.data["cERR"],
            ysig=data.data["x1_standardized_ERR"],
        )
        lm.run_mcmc(
            miniter=LINMIX_MIN_ITR
        )  # can do  silent=(not VERBOSE) once up and running.
        print(
            f"Beta = {np.median(lm.chain['beta']):.3f} +/- {robust_scatter(lm.chain['beta']):.3f}\n"
        )

    # Work with sim data
    ####
    BS21 = Fitres(Path("data/COMBINED_SIMS.FITRES"), alpha)
    BS21.clean_data(x1err_max, x1_max, cerr_max, c_min)
    BS21.calc_HR()

    G10 = Fitres(Path("data/G10_SIMDATA.FITRES"), alpha)
    G10.clean_data(x1err_max, x1_max, cerr_max, c_min)
    G10.calc_HR()

    C11 = Fitres(Path("data/C11_SIMDATA.FITRES"), alpha)
    C11.clean_data(x1err_max, x1_max, cerr_max, c_min)
    C11.calc_HR()

    # Plots
    ###
    for c_split in c_splits:
        data.slipt_on_c(c_split)
        # data.plot_hists("FITPROB", f"fitprob_dist_{c_split}.pdf")  # Not in paper
        data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
        data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")

    # data.plot_fitprob_c()   # Not in paper
    data.plot_fitprob_binned_c()

    data.plot_hist("c", f"c_dist.pdf")
    # data.plot_hist_c_special("c", f"c_dist_special.pdf")   # Not in paper

    plot_binned(
        data.data.loc[data.data["HOST_LOGMASS"] > 10],
        BS21.data.loc[BS21.data["HOST_LOGMASS"] > 10],
        "c",
        "x1_standardized",
        filename="color-luminosity-high_mass.png",
        fig_options={
            "data_name": "High Mass Host",
            "sim_name": "BS21",
            # "y_label": "mB - mu(z) - 0.15 * x1",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
        },
    )
    plot_binned(
        data.data.loc[data.data["HOST_LOGMASS"] <= 10],
        BS21.data.loc[BS21.data["HOST_LOGMASS"] <= 10],
        "c",
        "x1_standardized",
        filename="color-luminosity-low_mass.png",
        fig_options={
            "data_name": "Low Mass Host",
            "sim_name": "BS21",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
        },
    )
    plot_binned(
        data.data,
        BS21.data,
        "c",
        "HOST_LOGMASS",
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
        fit=lm.chain,
        filename="color-luminosity-BS21.png",
        fig_options={
            "sim_name": "BS21",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
        },
    )
    plot_binned(
        data.data,
        G10.data,
        "c",
        "x1_standardized",
        filename="color-luminosity-G10.png",
        fig_options={
            "sim_name": "G10",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
        },
    )
    plot_binned(
        data.data,
        C11.data,
        "c",
        "x1_standardized",
        filename="color-luminosity-C11.png",
        fig_options={
            "sim_name": "C11",
            "y_label": r"M$'$ (mag)",
            "y_flip": True,
        },
    )
