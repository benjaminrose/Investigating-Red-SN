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
        self.data = read_data(file_path, verbose)
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

    def cut_x1(self, x1_max):
        if self.VERBOSE:
            print(f"Initial x1 distribution:\n{self.data['x1'].describe()}\n")
        self.data = self.data[np.abs(self.data["x1"]) <= x1_max]
        if self.VERBOSE:
            print(f"After-cuts x1 distribution:\n{self.data['x1'].describe()}\n")
        return self.data

    def make_fitres_set(self, key="c", bin_size=None):
        """
        Make a set of SNANA FITRES files that split the dataset into bins.

        FITRES files are saved in a folder called "{key}_{bin_size}/".


        Parameters
        ----------
        key: string
            The FITRES key used to split on. Defaults to "c".
        bin_size: float
            The width of each FITRES bin. If `None` (default), just slits into three bins at c=0 and c=0.3.

        Returns
        -------
        Fitres class
        """
        if bin_size is None:
            save_location = Path(f"{key}_default/")
            save_location.mkdir(exist_ok=True)

            blue = self.data[self.data[key] <= 0]
            red = self.data[(0 < self.data[key]) & (self.data[key] <= 0.3)]
            very_red = self.data[0.3 < self.data[key]]

            for df, cut in zip([blue, red, very_red], [0, 0.3, 1]):
                self.to_file(save_location / f"PANTHEON_c_{cut}.FITRES", df)
        else:
            raise NotImplementedError
            save_location = Path(f"{key}_{bin_size}/")
            save_location.mkdir(exist_ok=True)

        cosmo_sample = self.data[(-0.3 < self.data[key]) & (self.data[key] <= 0.3)]
        self.to_file(save_location / f"PANTHEON_c_cosmo.FITRES", cosmo_sample)

        return self

    def plot_dit(self, key, filename=""):
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

    def plot_dits(self, key="FITPROB", filename=""):
        new_figure()

        ax = sns.histplot(
            data=self.blue_subsample,
            label=f"c <= {self.c_split}",
            x=key,
            color="b",
            **self.histplot_keywords,
        )
        sns.histplot(
            data=self.red_subsample,
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

    # def cut_c(self, c_min, c_max=10):
    #     if c_max != 10:
    #         print("c_max is not yet implimented.")
    #     if self.verbose:
    #         print(f"Initial c distribution:\n{self.data['c'].describe()}\n")
    #     self.data = self.data[self.data["c"] >= c_min]
    #     if self.verbose:
    #         print(f"After-cuts c distribution:\n{self.data['c'].describe()}\n")
    #     return self.data

    # def clean_data(self, x1_max, fitprob_min):
    #     """
    #     Parameters
    #     ----------
    #     c_min: float
    #         inclusive
    #     """
    #     self.data = self.cut_x1(self.data, x1_max)
    #     self.data = self.data[self.data["FITPROB"] > fitprob_min]
    #     return self.data

    # def calc_HR(self, M0, alpha, beta):
    #     """Adds columns "mu_theory", "x1_standardized", & "HR_naive"

    #     Parameters
    #     ----------
    #     data: pandas.DataFrame
    #         `data` needs columns "mB", "x1", "c". Data is assumed to be a
    #         DataFrame of an SNANA fitres file.
    #     M0:
    #     alpha:
    #     beta:
    #     """
    #     self.data["mu_theory"] = COSMO.distmod(self.data["zHD"]).value
    #     self.data["x1_standardized"] = (
    #         self.data["mB"] - self.data["mu_theory"] - alpha * self.data["x1"]
    #     )
    #     self.data["HR_naive"] = (
    #         self.data["mB"]
    #         + alpha * self.data["x1"]
    #         - beta * self.data["c"]
    #         + M0
    #         - self.data["mu_theory"]
    #     )

    #     if self.verbose:
    #         print(
    #             f"Naive distances:\n{self.data[['zHD', 'mu_theory', 'x1_standardized', 'HR_naive']].head(10)}\n"
    #         )
    #     return self.data

    # # def robust_scatter(x):
    # #     "aka converting from MAD to sigma equivalent"
    # #     return 1.4826 * np.median(np.abs(x - np.median(x)))

    # def measure_scatter(self):
    #     bins = np.arange(self.data["c"].min() - 0.01, self.data["c"].max() + 0.1, 0.1)

    #     HR_scatter, bin_edges, _ = binned_statistic(
    #         self.data["c"], self.data["HR_naive"], robust_scatter, bins
    #     )

    #     CL_scatter, _, _ = binned_statistic(
    #         self.data["c"],
    #         # self.data["mB"] - self.data["mu_theory"] - 0.15 * self.data["x1"],
    #         self.data["x1_standardized"],
    #         robust_scatter,
    #         bins,
    #     )
    #     if self.verbose:
    #         print("c_max, Scatter in HR, Scatter in Color-law plot")
    #         for bin, hl, cl in zip(bins[1:], HR_scatter, CL_scatter):
    #             print(f"{bin:.2f},\t{hl:.4f},\t\t\t{cl:.4f}")
    #         print("")

    #     return HR_scatter, CL_scatter, bins

    # def measure_scatter_by_mass(self):
    #     pass

    # def save_plot(self, filename, save):
    #     if save:
    #         folder = Path("figures/")
    #         folder.mkdir(exist_ok=True)
    #         plt.savefig("figures/" + filename, bbox_inches="tight")
    #     else:
    #         plt.show()

    # def plot_beta(self, M0, alpha, beta, save=True):
    #     _, ax = plt.subplots(tight_layout=True)

    #     # y = data["mB"] - data["mu_theory"] - alpha * data["x1"]
    #     x_main = np.linspace(self.data["c"].min(), self.data["c"].max(), 50)
    #     x_alt1 = np.linspace(0.15, 0.8, 50)
    #     x_alt2 = np.linspace(0.8, 1.0, 20)
    #     beta1 = 3.5
    #     M0_1 = 0.25
    #     beta2 = 1.0
    #     M0_2 = 2.0

    #     ax.plot(self.data["c"], self.data["x1_standardized"], ".")
    #     ax.plot(x_main, beta * x_main - M0, label=r"$\beta$" + f"={beta}, M0={M0}")
    #     ax.plot(
    #         x_alt1,
    #         beta1 * x_alt1 - M0 + M0_1,
    #         label=r"$\beta$" + f"={beta1}, M0={M0+M0_1}",
    #     )
    #     ax.plot(
    #         x_alt2,
    #         beta2 * x_alt2 - M0 + M0_2,
    #         label=r"$\beta$" + f"={beta2}, M0={M0+M0_2}",
    #     )

    #     ax.set(xlabel="Apparent Color c", ylabel=f"$ m_B - \mu(z)$ {alpha} $x_1$ (mag)")
    #     ax.invert_yaxis()
    #     plt.legend()

    #     save_plot("color-luminosity.pdf", save)

    # def plot_HR_c(self, save=True):
    #     _, ax = plt.subplots(tight_layout=True)

    #     ax.plot(self.data["c"], self.data["HR_naive"], ".")
    #     ax.set(xlabel="Apparent Color c", ylabel=f"HR")

    #     save_plot("color-HR.pdf", save)

    # def add_scatter_lines(self, ax, scatter, bins):
    #     plot_resolution = 10
    #     for i, scat in enumerate(scatter):
    #         x = np.linspace(bins[i], bins[i + 1], plot_resolution)
    #         ax.plot(x, [scat] * plot_resolution, "b")
    #     ax.set(xlabel="Apparent Color c", ylabel=f"Robust Scatter (mag)")
    #     return ax

    # def plot_scatter(self, HR_scatter, CL_scatter, bins, save=True):

    #     _, ax = plt.subplots(tight_layout=True)
    #     ax = add_scatter_lines(ax, HR_scatter, bins)
    #     ax.set(title="HR")
    #     save_plot("scatter_HR.pdf", save)

    #     _, ax = plt.subplots(tight_layout=True)
    #     ax = add_scatter_lines(ax, CL_scatter, bins)
    #     ax.set(title="Color-Luminosity Relation")
    #     save_plot("scatter_CL.pdf", save)

    # def plot_scatter_by_host(self, HR_scatter, CL_scatter, bins, save=True):
    #     pass
    #     # _, ax = plt.subplots(tight_layout=True)
    #     # ax = add_scatter_lines(ax, HR_scatter, bins)
    #     # ax = add_scatter_lines(ax, HR_scatter, bins)
    #     # save_plot("scatter_by_stellar_mass.pdf")

    # def plot_fitprob(self, save=True):
    #     _, ax = plt.subplots(tight_layout=True)
    #     ax.plot(self.data["c"], self.data["FITPROB"], ".")
    #     ax.set(xlabel="Apparent Color c", ylabel="SNANA Fit Probability")
    #     save_plot("color-fitprob.pdf", save)


if __name__ == "__main__":
    # Inputs
    data_file = Path("data") / Path("INPUT_FITOPT000.FITRES")
    VERBOSE = True
    c_splits = [0.1, 0.3]
    # c_min = 0.1
    x1_max = 3  # x1 cut is on abs(x1)
    # fitprob_min = 0.1
    # M0, alpha, beta = 19.34, 0.15, 3.1
    # COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)

    # Import and clean data
    data = Fitres(data_file, VERBOSE)
    data.cut_x1(x1_max)

    data.make_fitres_set()

    from sys import exit

    exit()

    for c_split in c_splits:
        data.slipt_on_c(c_split)
        data.plot_dits("FITPROB", f"fitprob_dist_{c_split}.pdf")
        data.plot_dits("x1", f"x1_dist_{c_split}.pdf")
        data.plot_dits("HOST_LOGMASS", f"mass_dist_{c_split}.pdf")
    data.plot_dit("c", f"c_dist.pdf")

    data.slipt_on_c(0.99)
    print("SN at c=1 boundry.")
    print(data.red_subsample[["CIDint", "IDSURVEY", "c"]])

    # OLd way
    # data = read_data(data_file)
    # data = clean_data(data, c_min, x1_max)
    # data = cut_c(data, c_min, x1_max)

    # # Manipulate data
    # data = calc_HR(data, M0, alpha, beta)

    # # Measure values
    # HR_scatter, CL_scatter, bins = measure_scatter(data)
    # # (
    # #     HR_scatter,
    # #     CL_scatter,
    # #     bins,
    # #     HR_scatter,
    # #     CL_scatter,
    # #     bins,
    # # ) = measure_scatter_by_mass(data)

    # # Make plots
    # plot_beta(data, M0, alpha, beta)
    # plot_HR_c(data)
    # plot_scatter(HR_scatter, CL_scatter, bins)
    # # plot_scatter_by_host(HR_scatter, CL_scatter, bins)
    # plot_fitprob(data)
