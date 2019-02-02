# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder, check_folders, separate_files

# Third-party module imports
import stats
from argparse import ArgumentParser
import logging

setup_logging()
logger = logging.getLogger("statistics")


def compute_statistics(wbb_dfs, fp_dfs):
    """Wrap all statistics computations."""

    logger.info("Computing general descriptive statistics.")


    domain_names = ["time_domain_features", "frequency_domain_features"]

    #################################
    # Dataframes HTML profile reports
    #################################

    logger.info("Generating profile reports.")

    stats.generate_all_profile_reports(wbb_dfs, fp_dfs, html_report_results_folder)

    ###########################################################
    # Features mean and standard deviations values computations
    ###########################################################

    logger.info("Computing mean and standard deviations values for each feature.")

    # Time features
    time_mean_and_std_results = stats.compute_mean_and_stds(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_mean_and_std_results)

    # Frequency features
    freq_mean_and_std_results = stats.compute_mean_and_stds(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_mean_and_std_results)

    ################
    # Paired T-test
    ################

    logger.info("Computing t-statistics and p-values for each feature.")

    # Time features
    time_t_test_results = stats.perform_t_test(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_t_test_results)

    # Frequency features
    freq_t_test_results = stats.perform_t_test(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_t_test_results)

    ######################
    # Spearman correlation
    ######################

    logger.info("Computing spearman correlation coefficients and p-values for each feature.")

    # Time features
    time_spearman_results = stats.compute_spearman_correlation(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_spearman_results)

    # Frequency features
    freq_spearman_results = stats.compute_spearman_correlation(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_spearman_results)

    ###########################################
    # Pearson correlation and linear regression
    ###########################################

    logger.info("Generating pearson correlation plots.")

    # Time features correlation plots
    time_correlation_results = stats.make_pearson_correlation_plots(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_correlation_results)

    # Frequency feature correlation plots
    freq_correlation_results = stats.make_pearson_correlation_plots(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_correlation_results)

    ##################################################################
    # Bland and Altman plots and Limits of Agreement(LOA) computations
    ##################################################################

    logger.info("Generating Bland and Altman agreement plots.")

    # Time features Bland and Altman plots
    time_loa = stats.make_bland_altman_plots(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_loa)

    # Frequency feature Bland and Altman plots
    freq_loa = stats.make_bland_altman_plots(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_loa)

    ########################################################
    # Intraclass Correlation Coefficients (ICC) computations
    ########################################################

    logger.info("Computing two-way mixed ICCs.")

    # Time features
    time_icc = stats.compute_ICC(wbb_dfs[0], fp_dfs[0], statistics_results_folders[0], domain_names[0])
    logger.debug(time_icc)

    # Frequency feature Bland and Altman plots
    freq_icc = stats.compute_ICC(wbb_dfs[1], fp_dfs[1], statistics_results_folders[1], domain_names[1])
    logger.debug(freq_icc)

    #########################
    # PUTTING IT ALL TOGETHER
    #########################

    logger.info("Statistical computations finished")

    ### TO DO


if __name__ == "__main__":

    ##################
    # Boilerplate code
    ##################

    # Load configuration files
    config = load_config()

    # Features computations results folder path
    feature_data_folder = config["feature_results_folder"]


    # Statistics results folder path
    html_report_results_folder = config["html_report_results_folder"]
    check_folder(html_report_results_folder)
    statistics_results_folders = [config["time_features_results_folder"], config["frequency_features_results_folder"]]
    check_folders(statistics_results_folders)

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-d", "--debug", action='store_true', help="Show debugging messages")
    args = parser.parse_args()
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    ###############
    # Data handling
    ###############

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(feature_data_folder)
    wbb_files_curated, fp_files_curated = separate_files(files)

    logger.info("Processing data located in: {}".format(feature_data_folder))

    # Create the pandas dataframes for the statistical analysis
    wbb_dfs = stats.construct_results_dfs(wbb_files_curated)
    fp_dfs = stats.construct_results_dfs(fp_files_curated)

    #########################
    # Statistics computations
    #########################
    compute_statistics(wbb_dfs, fp_dfs)
