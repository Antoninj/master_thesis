# Built-in modules imports
from pipeline import DataPipeline
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
from argparse import ArgumentParser
from tqdm import tqdm
import logging

setup_logging()
logger = logging.getLogger("feature extraction")


def process_all_wbb_files(data_pipeline, files):
    """Apply the frequency and time features extraction pipeline to the wii balance board data."""

    logger.info("Beginning of Wii Balance Board data processing")
    for file in tqdm(files):
        cop_data_pipeline.save_features(file, balance_board=True)

    logger.info("End of Wii Balance Board data processing")


def process_all_fp_files(data_pipeline, files):
    """Apply the frequency and time features extraction pipeline to the force plate data."""

    logger.info("Beginning of Force Plate data processing")
    for file in tqdm(files):
        cop_data_pipeline.save_features(file)

    logger.info("End of Force Plate data processing")


if __name__ == "__main__":

    # Load configuration files
    config = load_config()

    # Data folder path
    data_folder = config["data_folder"]

    # Results folder path
    results_folder = config["feature_results_folder"]
    check_folder(results_folder)

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Create a data pipeline object
    cop_data_pipeline = DataPipeline()

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(data_folder)

    # Separate WBB and force plate data
    wbb_files = [file for file in files if "Vicon" not in file]
    fp_files = [file for file in files if "Vicon" in file]

    logger.info("Executing feature extraction script.")
    logger.info("Processing data located in: {}".format(data_folder))

    if WBB:
        process_all_wbb_files(cop_data_pipeline, wbb_files)
    else:
        process_all_fp_files(cop_data_pipeline, fp_files)

    logger.info("Saving results to: {}".format(results_folder))
