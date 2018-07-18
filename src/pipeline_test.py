# Built-in modules imports
from pipeline import DataPipeline
from utils import load_config

# Third-party module imports
from argparse import ArgumentParser

if __name__ == "__main__":

    # Load configuration files
    config = load_config()

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

    if WBB:
        cop_data_pipeline.save_features(filepath_wbb, balance_board=True)
    else:
        cop_data_pipeline.save_features(filepath_fp)
