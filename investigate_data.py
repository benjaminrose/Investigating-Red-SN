""" investigate_red_sn.py

Requires python 3.9
"""
__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from astropy.cosmology import wCDM
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import binned_statistic, ks_2samp
import seaborn as sns

import linmix

from br_util.stats import robust_scatter
from br_util.snana import read_data
from br_util.plot import save_plot, new_figure

from KS2D.KS2D import ks2d2s

sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)


class Fitres:
    # should this subclass pandas?
    def __init__(self, file_path, alpha=0.15, verbose=False):
        self.data = read_data(file_path, 1, verbose)
        self.VERBOSE = verbose
        # Keywords used in plotting both data sets
        # So not: data, labels, colors, ...
        self.histplot_keywords = {
            "bins": "auto",
            "element": "step",
            "fill": False,
            "stat": "probability",
        }
        self.cumulative_plot_keywords = {
            **self.histplot_keywords,
            "cumulative": True,
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

    def clean_data(
        self, x1err_max=1.5, x1_max=3.0, cerr_max=0.15, c_min=-0.3, sim=False
    ):
        """Defulats follow Pantheon+.

        - x1min=-3.0
        - x1max=+3.0
        - ~~cmax=+0.3~~
        - cmin=-0.3
        - CUTWIN cERR 0 .15
        - CUTWIN x1ERR 0 1.5
        - CUTWIN PKMJDERR 0 20
        - CUTWIN MWEBV 0 .3
        - cutwin_Trestmin: -9999.0 5.0
        """
        # Apply Pantheon+ cleaning
        self._cut_x1err(x1err_max)
        self._cut_x1(x1_max)
        self._cut_cerr(cerr_max)
        self._cut_c(c_min)
        self._cut_PKMJDERR(20)  # checking once, no SN effected
        self._cut_MWEBV(0.3)  # checking once, ~25% of SN effected
        if not sim:
            self._cut_Trestmin(5.0)  # checking once, no SN effected

        # Also fix mass issues
        self._clean_mass()
        return self

    def _clean_mass(self):
        """
        These objects have crazy low masses.
        CID
        1290127    2.000
        1266657    2.000

        There is one other < 10^7 stellar mass value, but 10^7.0 seems
        to be the "not detected" value.

        Shift all masses that are < 10^6 to 10^7.
        """
        TOO_LOW_MASS = 6.0
        MASS_NOT_DETECTED_VALUE = 7.0

        if self.VERBOSE:
            print(
                f'SN with low stellar mass, pre-cleaning:\n{self.data.loc[self.data["HOST_LOGMASS"] < 7.0, "HOST_LOGMASS"]}'
            )

        self.data.loc[
            self.data["HOST_LOGMASS"] < TOO_LOW_MASS, "HOST_LOGMASS"
        ] = MASS_NOT_DETECTED_VALUE

        if self.VERBOSE:
            print(
                f'SN with low stellar mass, post-cleaning:\n{self.data.loc[self.data["HOST_LOGMASS"] < 7.0, "HOST_LOGMASS"]}'
            )

    def _cut_PKMJDERR(self, PKMJDERR):
        if self.VERBOSE:
            print("pre cut", self.data["PKMJDERR"].describe())
        self.data = self.data[self.data["PKMJDERR"] <= PKMJDERR]
        if self.VERBOSE:
            print("post cut", self.data["PKMJDERR"].describe())

    def _cut_MWEBV(self, MWEBV):
        if self.VERBOSE:
            print("pre cut", self.data["MWEBV"].describe())
        self.data = self.data[self.data["MWEBV"] <= MWEBV]
        if self.VERBOSE:
            print("post cut", self.data["MWEBV"].describe())

    def _cut_Trestmin(self, TrestMIN):
        if self.VERBOSE:
            print("pre cut", self.data["TrestMIN"].describe())
        self.data = self.data[self.data["TrestMIN"] <= TrestMIN]
        if self.VERBOSE:
            print("post cut", self.data["TrestMIN"].describe())

    def _cut_c(self, c_min):
        if self.VERBOSE:
            print(f"Initial c distribution:\n{self.data[['c', 'cERR']].describe()}\n")
        self.data = self.data[c_min <= self.data["c"]]
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

        sub_set1 = self.blue_subsample[
            ~self.blue_subsample.index.duplicated(keep="first")
        ]
        sub_set2 = self.red_subsample[
            ~self.red_subsample.index.duplicated(keep="first")
        ]

        ax = sns.histplot(
            data=sub_set1,
            label=f"c <= {self.c_split}",
            x=key,
            color="b",
            **self.histplot_keywords,
        )
        sns.histplot(
            data=sub_set2,
            x=key,
            ax=ax,
            label=f"c > {self.c_split}",
            color="r",
            **self.histplot_keywords,
        )
        plt.legend()

        print(
            f"KS test (two-sided) p-value for {filename}: {ks_2samp(sub_set1[key], sub_set2[key])[1]}."
        )

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
    COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)

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
