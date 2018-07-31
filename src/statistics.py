# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
import pandas as pd
import pandas_profiling
import numpy as np
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


def generate_profile_report(df, filename, bins=50):
    """
    Create a HTML profile report of a dataframe using general descriptive statistics.

    The profile report is generated using the pandas profiling (https://github.com/pandas-profiling) library.
    """

    # Create the report
    df_profile = pandas_profiling.ProfileReport(df, bins=bins)

    # Save the report
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


def plot_correlation(df1, df2, name="time_domain_features"):
    """Perform a linear least-squares regression and plot the correlation line for each feature."""

    fig, axs = plt.subplots(8, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs[-1, -1].axis('off')

    result_dict = {}
    # Loop over each feature
    for ax, column in zip(axs.ravel(), df1.columns):
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Perform the linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Store the R and p-value results
        result_dict[column] = {}
        result_dict[column]["R"] = r_value
        result_dict[column]["p-value"] = p_value

        # Make the plot
        ax.plot(x, y, '.', label='original data')
        ax.plot(x, intercept + slope * x, 'black', label='fitted line', linewidth=0.3)
        ax.set_xlabel('Balance Board')
        ax.set_ylabel('Force plate')
        ax.set_title(column, weight=600)
        r_squared = round(r_value**2, 4)
        ax.text(0.9, 0.5, "R\u00b2={}".format(r_squared), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # ax.legend()

    # Save the plots
    plt.savefig("{}/{}_correlation_plots.png".format(statistics_results_folder, name), bbox_inches='tight')

    return result_dict


def bland_altman_plot(df1, df2, name="time_domain"):

    fig, axs = plt.subplots(8, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs[-1, -1].axis('off')

    result_dict = {}
    # Loop over each feature
    for ax, column in zip(axs.ravel(), df1.columns):
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Prepare the data for the plots
        mean = np.mean([x, y], axis=0)
        diff = x - y
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        result_dict[column] = {}
        result_dict[column]["LOA"] = "{},{}".format(md - 2 * sd, md + 2 * sd)

        # Make the plot
        ax.scatter(mean, diff, marker='.', s=100, color="gray")
        ax.axhline(md, color='tomato', linestyle='--')
        ax.axhline(md + 2 * sd, color='teal', linestyle='--')
        ax.axhline(md - 2 * sd, color='teal', linestyle='--')
        ax.set_xlabel('Mean of the two systems')
        ax.set_ylabel('Mean of the difference')
        ax.set_title(column, weight=600)

        # ax.legend()

    # Save the plots
    plt.savefig("{}/{}_bland_altman_plots.png".format(statistics_results_folder, name), bbox_inches='tight')

    return result_dict


if __name__ == "__main__":

    ##################
    # Boilerplate code
    ##################

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
    parser.add_argument("-d", "--debug", action='store_true', help="Show debug messages")
    args = parser.parse_args()
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    ###############
    # Data handling
    ###############

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

    #################################
    # Dataframes HTML profile reports
    #################################

    logger.info("Generating profile reports.")
    generate_all_profile_reports(wbb_dfs, fp_dfs)

    ###########################################################
    # Features mean and standard deviations values computations
    ###########################################################

    logger.info("Computing mean and standard deviations values for each feature.")
    time_domain_results = compute_mean_and_stds(wbb_dfs[0], fp_dfs[0])
    freq_domain_results = compute_mean_and_stds(wbb_dfs[1], fp_dfs[1])

    ###################
    # Correlation plots
    ###################

    logger.info("Generating correlation plots.")

    # Time features correlation plots
    time_correlation_results = plot_correlation(wbb_dfs[0], fp_dfs[0])
    logger.debug(time_correlation_results)

    # Frequency feature correlation plots
    freq_correlation_results = plot_correlation(wbb_dfs[1], fp_dfs[1], name="frequency_domain_features")
    logger.debug(freq_correlation_results)

    ##################################################################
    # Bland and Altman plots and Limits of Agreement(LOA) computations
    ##################################################################

    logger.info("Generating Bland and Altman agreement plots.")

    # Time features Bland and Altman plots
    time_loa = bland_altman_plot(wbb_dfs[0], fp_dfs[0])
    logger.debug(time_loa)

    # Frequency feature Bland and Altman plots
    freq_loa = bland_altman_plot(wbb_dfs[1], fp_dfs[1], name="frequency_domain_features")
    logger.debug(time_loa)

    ########################################################
    # Intraclass Correlation Coefficients (ICC) computations
    ########################################################

    #########################
    # PUTTING IT ALL TOGETHER
    #########################

    logger.info("Saving results to: {}".format(statistics_results_folder))
