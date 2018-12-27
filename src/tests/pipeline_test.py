# Third-party module imports
import logging
from argparse import ArgumentParser

from context import *

if __name__ == "__main__":

    ##################
    # Boilerplate code
    ##################

    # Load configuration files
    config = load_config()

    # Setup logger
    setup_logging()
    logger = logging.getLogger("tests")

    # WBB data test file
    wbb_raw_data = config["test_files"]["wbb_raw_data"]

    # Force plate data test file
    fp_raw_data = config["test_files"]["fp_raw_data"]

    # WBB data test file
    wbb_cop_data = config["test_files"]["wbb_cop_data"]

    # FP data test file
    fp_cop_data = config["test_files"]["fp_cop_data"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    parser.add_argument("-c", "--cop", action='store_true', help="Save cop data")
    args = parser.parse_args()
    WBB = args.wbb
    COP = args.cop

    # Create a data pipeline object
    data_pipeline = DataPipeline()

    ##################
    # Tests
    ##################

    logger.info("Testing data pipeline module.")

    if WBB:
        logger.info("Test file: {}".format(wbb_raw_data))

        logger.info("Pre-processing Wii Balance Board acquistion data.")
        data_pipeline.preprocess_acquisition_file(wbb_raw_data, balance_board=WBB)

        logger.info("Processing Wii Balance Board COP data.")
        data_pipeline.process_cop_data_file(wbb_cop_data, balance_board=WBB)

    else:
        logger.info("Test file: {}".format(fp_raw_data))

        logger.info("Pre-processing Force Plate acquisition data.")
        data_pipeline.preprocess_acquisition_file(fp_raw_data)

        logger.info("Processing Force Plate COP data.")
        data_pipeline.process_cop_data_file(fp_cop_data)
