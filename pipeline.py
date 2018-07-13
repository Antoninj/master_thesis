from sensor import SensorDataReader
from preprocess import DataPreprocessor
from time_features import TimeFeatures
from frequency_features import FrequencyFeatures
from cop import *
from utils import save_as_json


class DataPipeline(object):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction
    """

    def __init__(self):
        self.data_reader = SensorDataReader()
        self.data_preprocessor = DataPreprocessor()

    def compute_cop_positions(self, balance_board=False):
        """
        Function to read the acquisition file raw data and compute the COP positions in the AP and ML directions
        """

        raw_data = self.data_reader.get_raw_data(balance_board)
        cop_wbb_x = compute_cop_wbb_x(raw_data)
        cop_wbb_y = compute_cop_wbb_y(raw_data)
        cop_data = (cop_wbb_x, cop_wbb_y)

        return cop_data

    def preprocess_cop_positions(self, cop_data, frequency, balance_board=False):
        """
        Function to preprocess the COP positions in the AP and ML directions and store them as a dictionarry
        """

        labels = ["COP_x", "COP_y"]
        preprocessed_data = [self.data_preprocessor.preprocess(data, frequency, balance_board) for data in cop_data]

        return dict(zip(labels, preprocessed_data))

    def save_cop_positions(self, filepath, balance_board=False):
        """
        Wrapper function that pipelines the COP computations and preprocessing steps and saves the results to a json file
        """

        self.data_reader.set_reader_filename(filepath)

        # Compute COP positions
        cop_positions = self.compute_cop_positions(balance_board)

        # Preprocess COP position
        frequency = config["preprocessing_parameters"]["acquisition_frequency"]
        preprocessed_cop_positions = self.preprocess_cop_positions(cop_positions, frequency, balance_board)

        # Save cop data
        save_as_json(preprocessed_cop_positions, filepath, "cop")

    def compute_time_features(self, cop_x, cop_y):
        """ Function to retrieve the time domain features """

        time_features = TimeFeatures(cop_x, cop_y)

        return time_features.time_features

    def compute_frequency_features(self, cop_x, cop_y):
        """ Function to retrieve the frequency domain features """

        frequency_features = FrequencyFeatures(cop_x, cop_y)

        return frequency_features.frequency_features

    def save_features(self, filepath, balance_board=False):
        """
        Wrapper function that pipelines the COP computations, preprocessing and feature extraction steps and saves the results to a json file
        """

        self.data_reader.set_reader_filename(filepath)

        # Compute COP positions
        cop_positions = self.compute_cop_positions(balance_board)

        # Preprocess COP position
        frequency = config["preprocessing_parameters"]["acquisition_frequency"]
        preprocessed_cop_positions = self.preprocess_cop_positions(cop_positions, frequency, balance_board)

        # Compute features
        time_features = self.compute_time_features(preprocessed_cop_positions["COP_x"], preprocessed_cop_positions["COP_y"])
        frequency_features = self.compute_frequency_features(preprocessed_cop_positions["COP_x"], preprocessed_cop_positions["COP_y"])

        merged_features = {"time_features": time_features, "frequency_features": frequency_features}

        # Save features
        save_as_json(merged_features, filepath, "features")
