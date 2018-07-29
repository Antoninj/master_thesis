# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
import pandas as pd
import pandas_profiling
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import logging
import json

setup_logging()
logger = logging.getLogger("stats")


def construct_results_dfs(files):
    """Aggregate all time and frequency feature computations results in dataframes."""

    time_frames = []
    frequency_frames = []
    for filepath in files:
        with open(filepath) as json_data:
            features = json.load(json_data)
            time_features = features["time_features"]
            frequency_features = features["frequency_features"]
        time_frames.append(pd.DataFrame(time_features, index=[0]))
        frequency_frames.append(pd.DataFrame(frequency_features, index=[0]))

    time_features_df = pd.concat(time_frames, axis=0)
    frequency_features_df = pd.concat(frequency_frames, axis=0)

    df1 = time_features_df.reset_index().drop('index', 1)
    df2 = frequency_features_df.reset_index().drop('index', 1)

    return [df1, df2]


def generate_profile_report(df, filename):
    """Create a HTML profile report of a dataframe using general descriptive statistics.

    The profile report is generated using the pandas profiling (https://github.com/pandas-profiling) library."""

    df_profile = pandas_profiling.ProfileReport(df, bins=50)
    df_profile.to_file(outputfile=filename)


def generate_all_profile_reports(wbb_dataframes, fp_dataframes):
    """Create all the profile reports."""

    domain_names = ["time_domain_features", "frequency_domain_features"]
    wbb_report_names = ["{}/wbb_{}_report.html".format(statistics_results_folder, name) for name in domain_names]
    fp_report_names = ["{}/fp_{}_report.html".format(statistics_results_folder, name) for name in domain_names]

    dfs = wbb_dfs + fp_dfs
    report_names = wbb_report_names + fp_report_names
    for (data, name) in zip(dfs, report_names):
        generate_profile_report(data, name)


def compute_mean_and_stds(df1, df2):
    """Compute the mean and standard deviation values for each feature."""

    wbb_mean_df = pd.DataFrame(df1.mean(), columns=["WBB mean"])
    wbb_std_df = pd.DataFrame(df1.std(), columns=["WBB std"])
    fp_mean_df = pd.DataFrame(df2.mean(), columns=["FP mean"])
    fp_std_df = pd.DataFrame(df2.std(), columns=["FP std"])

    aggregated_results = pd.concat([wbb_mean_df, wbb_std_df, fp_mean_df, fp_std_df], axis=1)

    return aggregated_results


def plot_feature_correlation(df1, df2, name="time_domain_features"):
    """Perform a linear least-squares regression and plot the correlation line for each feature."""

    columns = df1.columns
    fig, axs = plt.subplots(8, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs[-1, -1].axis('off')
    for ax, column in zip(axs.ravel(), columns):
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax.plot(x, y, '.', label='original data')
        ax.plot(x, intercept + slope * x, 'black', label='fitted line', linewidth=0.3)
        ax.set_xlabel('Balance Board')
        ax.set_ylabel('Force plate')
        ax.set_title(column, weight=600)
        r_squared = round(r_value**2, 4)
        ax.text(0.9, 0.5, "R\u00b2={}".format(r_squared), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # ax.legend()
    plt.savefig("{}/{}_correlation_plots.png".format(statistics_results_folder, name), bbox_inches='tight')


if __name__ == "__main__":

    # Load configuration files
    config = load_config()

    # Features computations results folder path
    feature_data_folder = config["feature_results_folder"]

    # Statistics results folder path
    statistics_results_folder = config["statistics_results_folder"]
    check_folder(statistics_results_folder)

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(feature_data_folder)

    # Separate WBB and force plate data
    wbb_files = [file for file in files if "Vicon" not in file and "cop" not in file]
    fp_files = [file for file in files if "Vicon" in file and "cop" not in file]

    logger.info("Processing data located in: {}".format(feature_data_folder))

    # Create the pandas dataframes for further analysis
    wbb_dfs = construct_results_dfs(wbb_files)
    fp_dfs = construct_results_dfs(fp_files)

    logger.info("Computing general descriptive statistics.")
    logger.info("Generating profile reports.")

    generate_all_profile_reports(wbb_dfs, fp_dfs)

    logger.info("Computing mean feature values and standard deviations.")

    time_domain_results = compute_mean_and_stds(wbb_dfs[0], fp_dfs[0])
    freq_domain_results = compute_mean_and_stds(wbb_dfs[1], fp_dfs[1])

    logger.info("Generating correlation plots.")

    # Time features correlation plots
    plot_feature_correlation(wbb_dfs[0], fp_dfs[0])

    # Frequency feature correlation plots
    plot_feature_correlation(wbb_dfs[1], fp_dfs[1], name="frequency_domain_features")

    logger.info("Generating Bland and Altman agreement plots.")

    logger.info("Saving results to: {}".format(statistics_results_folder))
