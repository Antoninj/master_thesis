# Built-in modules imports
from pipeline import DataPipeline
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
from argparse import ArgumentParser
import logging

setup_logging()
logger = logging.getLogger("feature extraction")


if __name__ == "__main__":

    ##################
    # Boilerplate code
    ##################

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

    ###############
    # Data handling
    ###############

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(data_folder)

    # Separate WBB and force plate data
    wbb_files = [file for file in files if "Vicon" not in file]
    fp_files = [file for file in files if "Vicon" in file]

    ####################
    # Feature extraction
    ####################

    logger.info("Executing feature extraction script.")
    logger.info("Processing data located in: {}".format(data_folder))

    # Create the pipeline object
    data_pipeline = DataPipeline()

    if WBB:
        logger.info("Beginning of Wii Balance Board data processing")

        # Assign WBB data to the pipeline object
        data_pipeline.set_pipeline_data(wbb_files)

        # Process all the WBB data
        data_pipeline.process_all_files(logger, balance_board=True)

        logger.info("End of Wii Balance Board data processing")

    else:
        logger.info("Beginning of Force Plate data processing")

        # Assign force plate data to the pipeline object
        data_pipeline.set_pipeline_data(fp_files)

        # Process all the force plate data
        data_pipeline.process_all_files()

        logger.info("End of Force Plate data processing")

    logger.info("Saving results to: {}".format(results_folder))
