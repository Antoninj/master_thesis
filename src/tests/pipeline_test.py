# Third-party module imports
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
    filepath_wbb = config["test_files"]["wbb_raw_data"]

    # Force plate data test file
    filepath_fp = config["test_files"]["fp_raw_data"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Create a data pipeline object
    cop_data_pipeline = DataPipeline()

    ##################
    # Tests
    ##################

    logger.info("Testing data pipeline module.")

    if WBB:
        logger.info("Processing Wii Balance Board data.")
        logger.info("Test file: {}".format(filepath_wbb))

        cop_data_pipeline.save_features(filepath_wbb, balance_board=True)
    else:
        logger.info("Processing Force Plate data.")
        logger.info("Test file: {}".format(filepath_fp))

        cop_data_pipeline.save_features(filepath_fp)
