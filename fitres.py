"""A class to import and clean up a fitres file
"""
from astropy.cosmology import wCDM
import numpy as np

from br_util.snana import read_data

# import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
from br_util.plot import save_plot, new_figure

from hypothesis_testing import ks1d

sns.set_theme(context="talk", style="ticks", font="serif", color_codes=True)


COSMO = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1)


class Fitres:
    # should this subclass pandas?
    def __init__(self, file_path, z_min, alpha=0.15, verbose=False):
        self.data = read_data(file_path, 1, verbose)
        self._cut_redshift(z_min)
        self.VERBOSE = verbose
        # Keywords used in plotting both data sets
        # So not: data, labels, colors, ...
        self.histplot_keywords = {
            "bins": "auto",
            "element": "step",
            "fill": False,
            "stat": "density",
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
            self.data["mB"]
            - self.data["mu_theory"]
            + np.abs(self.alpha) * self.data["x1"]
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
        self,
        x1err_max=1.5,
        x1_max=3.0,
        cerr_max=0.15,
        c_min=-0.3,
        sim=False,
        deduplicate=False,
    ):
        """Defulats follow Pantheon+.

        - x1min=-3.0
        - x1max=+3.0
        - ~~cmax=+0.3~~
        - cmin=-0.3
        - CUTWIN cERR 0 .15
        - CUTWIN x1ERR 0 1.5
        - CUTWIN PKMJDERR 0 2.0
        - CUTWIN MWEBV 0 .3
        - cutwin_Trestmin: -9999.0 5.0
        """
        if deduplicate:
            self._deduplicate()
        # Apply Pantheon+ cleaning
        self._cut_x1err(x1err_max)
        self._cut_x1(x1_max)
        self._cut_cerr(cerr_max)
        self._cut_c(c_min)
        self._cut_PKMJDERR(2.0)
        self._cut_MWEBV(0.3)  # checking once, ~25% of SN effected
        if not sim:
            self._cut_Trestmin(5.0)  # checking once, no SN effected
        self._cut_FITPROB(0)

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

    def _cut_FITPROB(self, min):
        if self.VERBOSE:
            print("pre cut", self.data["FITPROB"].describe())
        self.data = self.data[self.data["FITPROB"] > min]
        if self.VERBOSE:
            print("post cut", self.data["FITPROB"].describe())

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

    def _cut_redshift(self, min=0, max=100):
        self.data = self.data[min <= self.data["zHD"]]
        self.data = self.data[self.data["zHD"] <= max]
        return self

    def _deduplicate(self):
        duplicates = self.data[self.data.index.duplicated(keep=False)]
        singles = self.data[~self.data.index.duplicated(keep=False)]
        while 0 < len(duplicates):
            index = duplicates.index[0]
            if len(duplicates.loc[index, ["x1"]]) > 3:
                print("Ojbects with 4 surveys (plust 2006X):")
                print(duplicates.loc[index, ["IDSURVEY", "x1", "c", "x1ERR", "cERR"]])
            x1, c = duplicates.loc[index, ["x1", "c"]].mean()
            x1_err_min, c_err_min = (
                duplicates.loc[index, ["x1", "c"]].max()
                - duplicates.loc[index, ["x1", "c"]].min()
            )
            x1_err, c_err = duplicates.loc[index, ["x1ERR", "cERR"]].max()
            if x1_err < x1_err_min:
                x1_err = x1_err_min
            if c_err < c_err_min:
                c_err = c_err_min
            singles = singles.append(duplicates.loc[index].iloc[0])
            singles.loc[index, ["x1", "c", "x1ERR", "cERR"]] = x1, c, x1_err, c_err
            duplicates.drop(index, inplace=True)

        self.data = singles
        return self

    def plot_fitprob_binned_c(self, fitprob_cut=0.001):
        """
        'We follow past analyses of specific samples in order to employ
        a minimum Pfit cut: this is done for DES, PS1, and SDSS with
        levels of 0.01, 0.001, and 0.001, respectively.' - Scolnic+ 2022
        - Therefore, I will use fitprob_cut of 0.001 (not 0.01), but this
        does not change much.
        """
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

        _, ax = new_figure()

        ax.errorbar(
            (data_edges[:-1] + data_edges[1:]) / 2,
            data_stat * 100,  # Make a percent
            fmt=".",
            # label=f"Percent with FITPROB > {fitprob_cut}",
        )

        ax.set_xlabel("c")
        ax.set_ylabel("Percent pass")
        # ax.legend()
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
            fill=False,
            label="Pantheon+",
        )
        if key == "c":
            sns.rugplot(
                data=self.data[~self.data.index.duplicated(keep="first")],
                x=key,
                height=-0.02,
                lw=1,
                clip_on=False,
                ax=ax,
            )
            ax.fill(
                [-0.3, -0.3, 0.3, 0.3],
                [190, -10, -10, 190],
                facecolor="black",
                edgecolor="none",
                alpha=0.1,
            )
            ax.set_ylim(0, 185)
            sns.despine()
        if len(filename) > 0:
            save_plot(filename)
        else:
            return ax

    def plot_hist_plus(self, key, extra_data, filename=""):
        ax = self.plot_hist(key)

        # TODO: Don't hard code like this.
        labels = ["G10", "C11", "M11", "P22"]
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        line_styles = ["dotted", "dashed", "dashdot", (0, (3, 5, 1, 5, 1, 5))]
        for data, label, ls in zip(extra_data, labels, line_styles):
            ax = sns.histplot(
                # downsample non-duplicated sims to match self.data
                data=data[~data.index.duplicated(keep="first")].sample(len(self.data)),
                x=key,
                cumulative=False,
                bins="auto",
                element="step",
                fill=False,
                # stat="density",
                common_norm=True,
                label=label,
                zorder=1.5,  # move below data line that is at zorder=2.
                linestyle=ls,
                alpha=0.8,
            )
        ax.legend()

        save_plot(filename)

    def plot_hists(self, key="FITPROB", filename=""):
        new_figure()

        sub_set1 = self.blue_subsample[
            ~self.blue_subsample.index.duplicated(keep="first")
        ]
        sub_set2 = self.red_subsample[
            ~self.red_subsample.index.duplicated(keep="first")
        ]

        keywords = self.histplot_keywords
        if key == "zHD":
            keywords["stat"] = "proportion"

        ax = sns.histplot(
            data=sub_set1,
            label=r"c $\leq$" + f"{self.c_split}",
            x=key,
            color="b",
            **keywords,
        )
        sns.histplot(
            data=sub_set2,
            x=key,
            ax=ax,
            label=f"c > {self.c_split}",
            color="r",
            ls="--",
            **keywords,
        )
        if key == "x1":
            ax.set_xlabel(r"$x_1$")
        elif key == "HOST_LOGMASS":
            ax.set_xlabel(r"$\log(M_{*}/M_{\odot})$")
        elif key == "zHD":
            ax.set_xlabel("Redshift")
        ax.legend()

        if key == "zHD":
            ax.set_xlim(0.0, 1.0)

        ax.axes.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticks([])
        # ax.axes.yaxis.set_visible(False) # even remvoes label.
        ax.set_ylabel("Relative Count")  # I know it is a lie, but no one cares.
        sns.despine()

        print(
            f"KS test (two-sided) p-value for {filename}: {ks1d(sub_set1[key], sub_set2[key])}."
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
