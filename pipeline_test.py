from pipeline import DataPipeline
from cop import *
import argparse

if __name__ == "__main__":

    # Load configuration files
    config = load_config()

    # WBB data test file
    filepath_wbb = config["test_files"]["wbb_raw_data"]

    # Force plate data test file
    filepath_fp = config["test_files"]["fp_raw_data"]

    # Command line argument parser to choose between wbb or force plate data
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Create a data pipeline object
    cop_data_pipeline = DataPipeline()

    if WBB:
        cop_data_pipeline.save_wbb_cop_positions(filepath_wbb)
    else:
        cop_data_pipeline.save_fp_cop_positions(filepath_fp)
