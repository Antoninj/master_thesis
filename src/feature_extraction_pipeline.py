# Built-in modules imports
from pipeline import DataPipeline
from utils import load_config, get_path_to_all_files, setup_logging, check_folder

# Third-party module imports
from argparse import ArgumentParser
import logging

setup_logging()
logger = logging.getLogger("feature extraction pipeline")


if __name__ == "__main__":

    ##################
    # Boilerplate code
    ##################

    # Load configuration file
    config = load_config()

    # Data folder path
    data_folder = config["cop_data_folder"]

    # Results folder path
    results_folder = config["feature_results_folder"]
    check_folder(results_folder)

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    parser.add_argument("-d", "--debug", action='store_true', help="Enable debug mode")

    args = parser.parse_args()
    WBB = args.wbb
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    ################
    # Files handling
    ################

    # Get all the paths to the files that need to be processed
    files = get_path_to_all_files(data_folder)

    # Separate WBB and force plate data
    wbb_files = [file for file in files if "FP" not in file]

    fp_files = [file for file in files if "FP" in file and "RSSCAN" not in file]

    ####################
    # Feature extraction
    ####################

    logger.info("Executing feature extraction pipeline.")
    logger.info("Processing COP data located in: {}".format(data_folder))

    # Create the pipeline object
    data_pipeline = DataPipeline()

    if WBB:
        logger.info("Beginning of Wii Balance Board COP data processing")

        # Assign WBB data to the pipeline object
        data_pipeline.set_pipeline_cop_data(wbb_files)

        # Process all the WBB data
        data_pipeline.process_all_files(logger, balance_board=True)

        logger.info("End of Wii Balance Board COP data processing")

    else:
        logger.info("Beginning of Force Plate COP data processing")

        # Assign force plate data to the pipeline object
        data_pipeline.set_pipeline_cop_data(fp_files)

        # Process all the force plate data
        data_pipeline.process_all_files(logger)

        logger.info("End of Force Plate COP data processing")

    logger.info("Saving results to: {}".format(results_folder))
