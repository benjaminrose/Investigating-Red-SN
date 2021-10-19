"""Utility functions"""

__version__ = "2021-09"

from argparse import ArgumentParser, BooleanOptionalAction

from scipy.stats import binned_statistic

from br_util.stats import robust_scatter


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


def parse_cli():
    arg_parser = ArgumentParser(description=__doc__)
    arg_parser.add_argument("--version", action="version", version=__version__)
    arg_parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="number of bins to use in color-luminosity plot (default: %(default)s)",
    )
    # arg_parser.add_argument(
    #     "--cmax",
    #     type=float,
    #     default=2.0,
    #     help="maximum c used in fitting BETA (default: %(default)s)",
    # )
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
