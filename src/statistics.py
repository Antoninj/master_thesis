# Built-in modules imports
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
from argparse import ArgumentParser
import pandas as pd
import pandas_profiling
import logging
import json

setup_logging()
logger = logging.getLogger("stats")


def construct_results_dfs(files):
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


def generate_html_report(df, filename):

    df_profile = pandas_profiling.ProfileReport(df, bins=50)
    df_profile.to_file(outputfile=filename)


def generate_all_html_reports(wbb_files, fp_files):
    wbb_dfs = construct_results_dfs(wbb_files)
    fp_dfs = construct_results_dfs(fp_files)

    domain_names = ["time_domain", "freq_domain"]
    wbb_report_names = ["{}/wbb_{}_report.html".format(statistics_results_folder, name) for name in domain_names]
    fp_report_names = ["{}/fp_{}_report.html".format(statistics_results_folder, name) for name in domain_names]

    dfs = wbb_dfs + fp_dfs
    report_names = wbb_report_names + fp_report_names
    for (data, name) in zip(dfs, report_names):
        generate_html_report(data, name)


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

    # Get all the filepaths to the files that need to be processed
    files = get_path_to_all_files(feature_data_folder)

    # Separate WBB and force plate data
    wbb_files = [file for file in files if "Vicon" not in file and "cop" not in file]
    fp_files = [file for file in files if "Vicon" in file and "cop" not in file]

    logger.info("Descriptive statistics generation script.")
    logger.info("Processing data located in: {}".format(feature_data_folder))

    logger.info("Generating HTML reports.")

    generate_all_html_reports(wbb_files, fp_files)

    logger.info("Storing results in: {}".format(statistics_results_folder))
