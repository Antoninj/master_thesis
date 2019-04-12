# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder, check_folders, separate_files

# Third-party module imports
import stats
from argparse import ArgumentParser
import logging
import timeit

setup_logging()
logger = logging.getLogger("statistical analysis")


def compute_time_features_stats():
    compute_statistics(fp_time_feature_df, wbb_time_feature_df, statistics_results_folders[0],
                       html_report_results_folders[0], html_report_results_folders[1])


def compute_frequency_features_stats():
    compute_statistics(fp_frequency_feature_df, wbb_frequency_feature_df, statistics_results_folders[1],
                       html_report_results_folders[2], html_report_results_folders[3], plot_size=6)


def compute_all_statistics():
    ######################################
    # Time feature statistics computations
    ######################################

    logger.info("Computing time features statistics...")

    compute_time_features_stats()

    ###########################################
    # Frequency feature statistics computations
    ###########################################

    logger.info("Computing frequency features statistics...")
    compute_frequency_features_stats()

    logger.info("Statistical computations finished!")


def compute_statistics(fp_df, wbb_df, statistics_results_folder, wbb_html_report_results_folder,
                       fp_html_report_results_folder, plot_size=7):
    """Wrap all statistics computations."""

    logger.info("Computing general descriptive statistics.")

    #################################
    # Dataframes HTML profile reports
    #################################

    logger.info("Generating profile reports using pandas profiling.")

    fp_profile_report_filename = "{}/FP_report.html".format(fp_html_report_results_folder)
    wbb_profile_report_filename = "{}/WBB_report.html".format(wbb_html_report_results_folder)

    # Aggregated WBB data granularity level
    stats.generate_profile_report(fp_df, fp_profile_report_filename)
    stats.generate_profile_report(wbb_df, wbb_profile_report_filename)

    # Individual Balance Board data granularity level
    stats.generate_all_profile_reports(fp_df, fp_html_report_results_folder)
    stats.generate_all_profile_reports(wbb_df, wbb_html_report_results_folder)

    ###########################################################
    # Features mean and standard deviations values computations
    ###########################################################

    logger.info("Computing mean and standard deviations values for each feature.")

    mean_and_std_results = stats.compute_mean_and_stds(wbb_df, fp_df, statistics_results_folder)
    logger.debug(mean_and_std_results)


    ######################
    # Spearman correlation
    ######################

    logger.info("Computing spearman correlation coefficients and p-values.")

    spearman_results = stats.compute_spearman_correlation(fp_df, wbb_df, statistics_results_folder)
    logger.debug(spearman_results)

    ###########################################
    # Pearson correlation and linear regression
    ###########################################

    logger.info("Generating regression plots using ODR.")

    # Balance Board granularity level
    linear_regression_results = stats.make_pearson_correlation_plots(fp_df, wbb_df, statistics_results_folder,
                                                                     plot_size)
    logger.debug(linear_regression_results)

    # Balance Board aggregated data
    global_linear_regression_results = stats.make_global_person_correlation_plots(fp_df, wbb_df,
                                                                                  statistics_results_folder,
                                                                                  plot_size)
    logger.debug(global_linear_regression_results)

    ##################################################################
    # Bland and Altman plots and Limits of Agreement(LOA) computations
    ##################################################################

    logger.info("Generating Bland and Altman agreement plots.")

    bland_altman = stats.make_bland_altman_plots(fp_df, wbb_df, statistics_results_folder, plot_size)
    logger.debug(bland_altman)

    ########################################################
    # Intraclass Correlation Coefficients (ICC) computations
    ########################################################

    logger.info("Computing two-way mixed ICCs.")

    icc_results_1 = stats.compute_ICC(wbb_df, statistics_results_folder)
    icc_results_2 = stats.compute_ICC_2(wbb_df, statistics_results_folder)

    logger.debug(icc_results_2)


def get_duplicate_identity(data, balance_board_number):
    data = data.loc[data.index.get_level_values('balance board') == balance_board_number]

    return data[data.duplicated() == True].index.tolist()


def get_outlier_identity(data, feature_name, balance_board_number):
    data = data.loc[data.index.get_level_values('balance board') == balance_board_number]
    outlier_values = data[data[feature_name] > 4 * data[feature_name].std()]

    return outlier_values.index.tolist()


if __name__ == "__main__":

    start = timeit.default_timer()

    ##################
    # Boilerplate code
    ##################

    # Load configuration file
    config = load_config()

    # Features computations results folder path
    feature_data_folder = config["feature_results_folder"]

    # Statistics results folder path
    html_report_results_folders = config["html_report_results_folders"]
    check_folders(html_report_results_folders)
    statistics_results_folders = [config["time_features_results_folder"], config["frequency_features_results_folder"]]
    check_folders(statistics_results_folders)

    # Command line argument parser: option to remove outliers from the study
    parser = ArgumentParser(
        description="")
    parser.add_argument("-d", "--debug", action='store_true', help="Show debugging messages")
    parser.add_argument('-o', "--outliers", action='store_false', help="remove outliers")
    args = parser.parse_args()
    debug = args.debug
    outliers = args.outliers

    if debug:
        logger.setLevel("DEBUG")

    ###############
    # Data handling
    ###############

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(feature_data_folder)
    wbb_files_curated, fp_files_curated = separate_files(files)

    logger.info("Processing data located in: {}".format(feature_data_folder))

    # Create the pandas dataframes which contains all the aggregated data for the statistical analysis
    wbb_dfs = stats.construct_results_dfs(wbb_files_curated)
    wbb_time_feature_df = wbb_dfs[0]
    wbb_frequency_feature_df = wbb_dfs[1]

    fp_dfs = stats.construct_results_dfs(fp_files_curated)
    fp_time_feature_df = fp_dfs[0]
    fp_frequency_feature_df = fp_dfs[1]

    #####################
    # Duplicates removal
    #####################

    wbb_numbers = [str(i) for i in range(2, 4)]
    duplicate_indexes_1 = [get_duplicate_identity(fp_time_feature_df, wbb) for wbb in
                           wbb_numbers]
    duplicate_indexes_2 = [get_duplicate_identity(wbb_time_feature_df, wbb) for wbb in
                           wbb_numbers]

    logger.info("Removing duplicates: {}".format(duplicate_indexes_1))

    for df in [fp_time_feature_df, fp_frequency_feature_df]:
        [df.drop(index, inplace=True) for index in duplicate_indexes_1]

    for df in [wbb_time_feature_df, wbb_frequency_feature_df]:
        [df.drop(index, inplace=True) for index in duplicate_indexes_2]

    #############################
    # Outliers removal (optional)
    #############################
    if not args.outliers:
        outlier_feature = ["Range", "Range-AP", "Range-AP"]
        wbb_numbers = [str(i) for i in range(1, 4)]
        outlier_indexes_1 = [get_outlier_identity(fp_time_feature_df, id[0], id[1]) for id in
                             zip(outlier_feature, wbb_numbers)]
        outlier_indexes_2 = [get_outlier_identity(wbb_time_feature_df, id[0], id[1]) for id in
                             zip(outlier_feature, wbb_numbers)]

        logger.info("Removing outliers: {}".format(outlier_indexes_1))

        for df in [fp_time_feature_df, fp_frequency_feature_df]:
            [df.drop(index, inplace=True) for index in outlier_indexes_1]

        for df in [wbb_time_feature_df, wbb_frequency_feature_df]:
            [df.drop(index, inplace=True) for index in outlier_indexes_2]

    #########################
    # PUTTING IT ALL TOGETHER
    #########################
    compute_all_statistics()

    stop = timeit.default_timer()

    logger.info('Execution time: {} seconds'.format(stop - start))
