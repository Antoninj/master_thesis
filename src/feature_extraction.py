# Built-in modules imports
from pipeline import DataPipeline
from utils import load_config, get_path_to_all_files, setup_logging

# Third-party module imports
from argparse import ArgumentParser
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
setup_logging()


def process_all_wbb_files(data_pipeline, files):
    """Apply the frequency and time features extraction pipeline to the wii balance board data."""

    logger.info("Beginning of Wii Balance Board data processing")
    for file in tqdm(files):
        if "Vicon" not in file:
            cop_data_pipeline.save_features(file, balance_board=True)

    logger.info("End of Wii Balance Board data processing")


def process_all_fp_files(data_pipeline, files):
    """Apply the frequency and time features extraction pipeline to the force plate data."""

    logger.info("Beginning of Force Plate data processing")
    for file in tqdm(files):
        if "Vicon" in file:
            cop_data_pipeline.save_features(file)

    logger.info("End of Force Plate data processing")


if __name__ == "__main__":

    # Load configuration files
    config = load_config()

    # Data folder filepath
    data_folder = config["data_folder"]

    # Results folder filepath
    results_folder = config["results_folder"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Create a data pipeline object
    cop_data_pipeline = DataPipeline()

    # Get all the filepaths to the files that need to be processed
    filepaths = get_path_to_all_files(data_folder)

    if WBB:
        process_all_wbb_files(cop_data_pipeline, filepaths)
    else:
        process_all_fp_files(cop_data_pipeline, filepaths)
