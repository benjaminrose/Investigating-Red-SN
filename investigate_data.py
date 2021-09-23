""" investigate_red_sn.py
"""
from pathlib import Path
from re import VERBOSE

from astropy.cosmology import wCDM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import seaborn as sns

from br_util.stats import robust_scatter
from br_util.snana import read_data
from br_util.plot import save_plot, new_figure

sns.set(context="talk", style="ticks", font="serif", color_codes=True)


class Fitres:
    # should this subclass pandas?
    def __init__(self, file_path, verbose=False):
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
        self.alpha = 0.15
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

    def clean_data(self, x1_max=5, cerr_max=1):
        self._cut_x1(x1_max)
        self._cut_cerr(cerr_max)
        return self

    def _cut_cerr(self, cerr_max):
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

    def plot_hist(self, key, filename=""):
        new_figure()
        ax = sns.histplot(
            data=self.data[~self.data.index.duplicated(keep="first")],
            x=key,
            color="b",
            cumulative=False,
            # **self.histplot_keywords,
        )
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


def plot_beta(data, sim, filename=""):
    """
    Note
    ----
    The Fitres objects are not extentions to DataFrames (yet). To get
    to the DataFrame you need to use `Fitres.data`.
    """
    bins = 25

    fig, ax = new_figure()

    sim_mask = sim.data["c"] > -0.5
    sim_c = sim.data.loc[sim_mask, "c"]
    # sim_y_axis = sim.data.loc[sim_mask, "mB"]
    sim_y_axis = sim.data.loc[sim_mask, "x1_standardized"]
    data_x = data.data["c"]
    data_y = data.data["x1_standardized"]

    ax.plot(sim_c, sim_y_axis, ".", markersize=3, alpha=0.3, label="BS21 Simulations")
    ax.plot(data_x, data_y, ".", markersize=3, alpha=0.3, label="Pantheon+")

    sim_stat, sim_edges, _ = binned_statistic(
        sim_c, sim_y_axis, statistic="median", bins=bins
    )
    sim_error, _, _ = binned_statistic(
        sim_c, sim_y_axis, statistic=robust_scatter, bins=bins
    )
    data_stat, data_edges, _ = binned_statistic(
        data_x, data_y, statistic="median", bins=bins
    )
    data_error, _, _ = binned_statistic(
        data_x, data_y, statistic=robust_scatter, bins=bins
    )

    ax.errorbar(
        (sim_edges[:-1] + sim_edges[1:]) / 2,
        sim_stat,
        yerr=sim_error,
        fmt="^",
        label="Binned Simulations",
    )
    ax.errorbar(
        (data_edges[:-1] + data_edges[1:]) / 2,
        data_stat,
        yerr=data_error,
        fmt="^",
        label="Binned Data",
    )

    ax.set_xlabel("c")
    ax.set_ylabel("mB - mu(z) - 0.15 * x1")
    ax.legend(fontsize="small")

    save_plot(filename)


if __name__ == "__main__":
    # Inputs
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = True
    c_splits = [0.1, 0.3]
    x1_max = 3  # x1 cut is on abs(x1)
    cerr_max = 0.2
    # fitprob_min = 0.1
    COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)

    # Import and clean data
    data = Fitres(data_file, VERBOSE)
    data.clean_data(x1_max, cerr_max)
    data.calc_HR()

    data.slipt_on_c(0.99)
    print("SN at c=1 boundry.")
    print(data.red_subsample[["CIDint", "IDSURVEY", "c"]])

    for c_split in c_splits:
        data.slipt_on_c(c_split)
        data.plot_hists("FITPROB", f"fitprob_dist_{c_split}.pdf")
        data.plot_hists("x1", f"x1_dist_{c_split}.pdf")
        data.plot_hists("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")

    data.plot_hist("c", f"c_dist.pdf")
    data.plot_hist_c_special("c", f"c_dist_special.pdf")

    sims = Fitres(Path("data/COMBINED_SIMS.FITRES"))
    sims.clean_data(x1_max, cerr_max)
    sims.calc_HR()
    plot_beta(data, sims, "color-luminosity.png")
