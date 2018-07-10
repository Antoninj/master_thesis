from sensor import SensorDataReader
from preprocess import DataPreprocessor
import json
from cop import *
import argparse
from utils import load_config, NumpyEncoder
import os


class DataPipeline(object):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction
    """

    def __init__(self):
        self.data_reader = SensorDataReader()
        self.data_preprocessor = DataPreprocessor()

    def compute_wbb_cop_positions(self):
        """
        Function to read the WBB acquistion file raw data and compute the COP positions in the AP and ML directions
        """

        raw_data = self.data_reader.get_raw_data(balance_board=True)
        cop_wbb_x = compute_cop_wbb_x(raw_data)
        cop_wbb_y = compute_cop_wbb_y(raw_data)
        cop_data = (cop_wbb_x, cop_wbb_y)

        return cop_data

    def preprocess_wbb_cop_positions(self, cop_data, frequency):
        """
        Function to preprocess the WBB data COP positions in the AP and ML directions and store them as a dictionarry
        """

        labels = ["COP x", "COP y"]
        preprocessed_data = [self.data_preprocessor.preprocess(data, frequency, True) for data in cop_data]

        return dict(zip(labels, preprocessed_data))

    def save_wbb_cop_positions(self, filepath):
        """
        Wrapper function that pipelines the WBB data COP computations and preprocessing steps and saves the results to a json file
        """

        self.data_reader.set_reader_filename(filepath)
        wbb_cop_positions = self.compute_wbb_cop_positions()
        frequency = preprocessing_config["acquisition_frequency"]
        preprocessed_cop_positions = self.preprocess_wbb_cop_positions(wbb_cop_positions, frequency)

        base_image_name = os.path.splitext(filepath)[0]
        filename = base_image_name + "_cop.json"
        with open(filename, 'w') as outfile:
            json.dump(preprocessed_cop_positions, outfile, cls=NumpyEncoder, sort_keys=True, indent=4, ensure_ascii=False)

    def compute_fp_cop_positions(self):
        """
        Function to read the force plate acquistion file raw data and compute the COP positions in the AP and ML directions
        """

        raw_data = self.data_reader.get_raw_data()
        cop_fp_x = compute_cop_fp_x(raw_data)
        cop_fp_y = compute_cop_fp_y(raw_data)
        cop_data = (cop_fp_x, cop_fp_y)

        return cop_data

    def preprocess_fp_cop_positions(cop_data, frequency):
        """
        Function to preprocess the force plate data COP positions in the AP and ML directions and store them as a dictionarry
        """

        labels = ["COP x", "COP y"]
        preprocessed_data = [self.data_preprocessor.preprocess(data, frequency) for data in cop_data]

        return dict(zip(labels, preprocessed_data))

    def save_fp_cop_positions(self, filepath):
         """
        Wrapper function that pipelines the force plate data COP computations and preprocessing steps and saves the results to a json file
        """

        self.data_reader.set_reader_filename(filepath)
        fp_cop_positions = compute_fp_cop_positions(data_reader)
        analog_freq = data_reader.get_frequency()
        preprocessed_cop_positions = preprocess_fp_cop_positions(data_preprocessor, fp_cop_positions, analog_freq)

        base_image_name = os.path.splitext(filepath)[0]
        filename = base_image_name + "_cop.json"
        with open(filename, 'w') as outfile:
            json.dump(preprocessed_cop_positions, outfile, cls=NumpyEncoder, sort_keys=True, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    # Load configuration files
    test_config = load_config("test")
    preprocessing_config = load_config("preprocess")

    # WBB data test file
    filepath_wbb = test_config["wbb raw data test file"]

    # Force plate data test file
    filepath_fp = test_config["fp raw data test file"]


    # Command line argument parser to choose between wbb or force plate data
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--wbb", action='store_true', help="Process WBB data")
    args = parser.parse_args()
    WBB = args.wbb

    # Create a datapipeline object
    cop_data_pipeline = DataPipeline()

    if WBB:
        cop_data_pipeline.save_wbb_cop_positions(filepath_wbb)
    else:
        cop_data_pipeline.save_fp_cop_positions(filepath_fp)
