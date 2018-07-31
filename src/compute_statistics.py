# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
from stats import *
from argparse import ArgumentParser
import logging

setup_logging()
logger = logging.getLogger("stats")

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
    generate_all_profile_reports(wbb_dfs, fp_dfs, statistics_results_folder)

    ###########################################################
    # Features mean and standard deviations values computations
    ###########################################################

    logger.info("Computing mean and standard deviations values for each feature.")
    time_domain_results = compute_mean_and_stds(wbb_dfs[0], fp_dfs[0])
    freq_domain_results = compute_mean_and_stds(wbb_dfs[1], fp_dfs[1])

    ################
    # Paired T-test
    ################

    logger.info("Computing t-statistics and p-values for each feature.")
    # Time features
    time_t_test_results = compute_t_test(wbb_dfs[0], fp_dfs[0])
    logger.debug(time_t_test_results)

    # Frequency features
    freq_t_test_results = compute_t_test(wbb_dfs[1], fp_dfs[1])

    ######################
    # Spearman correlation
    ######################

    logger.info("Computing spearman correlation coefficients and p-values for each feature.")
    # Time features
    time_spearman_results = compute_spearman_correlation(wbb_dfs[0], fp_dfs[0])
    logger.debug(time_spearman_results)

    # Frequency features
    freq_spearman_results = compute_spearman_correlation(wbb_dfs[1], fp_dfs[1])

    ###########################################
    # Pearson correlation and linear regression
    ###########################################

    logger.info("Generating correlation plots.")
    # Time features correlation plots
    time_correlation_results = pearson_correlation_plots(wbb_dfs[0], fp_dfs[0], statistics_results_folder)
    logger.debug(time_correlation_results)

    # Frequency feature correlation plots
    freq_correlation_results = pearson_correlation_plots(wbb_dfs[1], fp_dfs[1], statistics_results_folder, name="frequency_domain_features")
    logger.debug(freq_correlation_results)

    ##################################################################
    # Bland and Altman plots and Limits of Agreement(LOA) computations
    ##################################################################

    logger.info("Generating Bland and Altman agreement plots.")
    # Time features Bland and Altman plots
    time_loa = bland_altman_plots(wbb_dfs[0], fp_dfs[0], statistics_results_folder)
    logger.debug(time_loa)

    # Frequency feature Bland and Altman plots
    freq_loa = bland_altman_plots(wbb_dfs[1], fp_dfs[1], statistics_results_folder, name="frequency_domain_features")
    logger.debug(time_loa)

    ########################################################
    # Intraclass Correlation Coefficients (ICC) computations
    ########################################################

    #########################
    # PUTTING IT ALL TOGETHER
    #########################

    logger.info("Saving results to: {}".format(statistics_results_folder))
