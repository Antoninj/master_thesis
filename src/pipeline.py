# Built-in modules imports
from sensor import SensorDataReader
from preprocess import DataPreprocessor
from time_features import TimeFeatures
from frequency_features import FrequencyFeatures
from cop import *
from utils import save_as_json
from tqdm import tqdm

# Third-party module imports
import logging

logger = logging.getLogger("pipeline")


class DataPipeline(SensorDataReader, DataPreprocessor):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction.
    """

    def __init__(self, files=None):
        super(DataPipeline, self).__init__()
        self.data = files

    def compute_cop_positions(self, preprocessed_data, balance_board=False):
        """Compute the COP positions in the AP and ML directions."""

        cop_data = {}
        try:
            if balance_board:
                cop_data["COP_x"] = compute_cop_wbb_x(preprocessed_data)
                cop_data["COP_y"] = compute_cop_wbb_y(preprocessed_data)
            else:
                cop_data["COP_x"] = compute_cop_fp_x(preprocessed_data)
                cop_data["COP_y"] = compute_cop_fp_y(preprocessed_data)

            return cop_data

        except Exception:
            raise

    def compute_time_features(self, cop_x, cop_y):
        """Compute the time domain features."""

        time_domain_features = TimeFeatures(cop_x, cop_y)

        return time_domain_features.time_features

    def compute_frequency_features(self, cop_x, cop_y):
        """Compute the frequency domain features."""

        frequency_domain_features = FrequencyFeatures(cop_x, cop_y)

        return frequency_domain_features.frequency_features

    def save_features(self, filepath, balance_board=False, save_cop=False):
        """
        Pipeline the COP computations, preprocessing and feature extraction steps and save the results to a json file."""

        try:
            # Get the raw data
            raw_data = self.get_raw_data(filepath, balance_board)

            # Preprocess the raw data
            frequency = config["preprocessing_parameters"]["acquisition_frequency"]
            preprocessed_data = self.preprocess_raw_data(raw_data, frequency, balance_board)

            # Compute the COP positions
            cop_data = self.compute_cop_positions(preprocessed_data, balance_board)

            # Preprocess the COP positions
            preprocessed_cop_data = self.preprocess_cop_data(cop_data)

            if save_cop:
                # Save intermediate results of COP computations
                save_as_json(preprocessed_cop_data, filepath, "cop_results", "cop")
                return

            # Compute time features from COP displacement
            time_features = self.compute_time_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            # Compute frequency features from COP displacement
            frequency_features = self.compute_frequency_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            merged_features = {"filepath": filepath, "time_features": time_features, "frequency_features": frequency_features}

            # Save features in json format
            save_as_json(merged_features, filepath, "feature_results", "features")

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def process_all_files(self, balance_board=False):
        """Write docstring"""

        for file in tqdm(self.data):
            self.save_features(file, balance_board)
