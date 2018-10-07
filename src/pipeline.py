# Built-in modules imports
from sensor import SensorDataReader
from preprocess import DataPreprocessor
from process import DataProcessor
from utils import save_as_json
from tqdm import tqdm
import sys

# Third-party module imports
import logging

logger = logging.getLogger("pipeline")


class DataPipeline(SensorDataReader, DataPreprocessor, DataProcessor):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction.
    """

    def __init__(self, files=None):
        super(DataPipeline, self).__init__()
        self.data = files

    def save_features(self, filepath, balance_board=False, save_cop=False):
        """
        Pipeline the preprocessing, COP computations and feature extraction steps and save the results to a json file."""

        try:
            # Get the raw data
            raw_data = self.get_raw_data(filepath, balance_board)

            # Preprocess the raw data
            preprocessed_data = self.preprocess_raw_data(raw_data, balance_board)

            # Compute the COP positions
            cop_data = self.compute_cop_positions(preprocessed_data, balance_board)

            # Detrend the COP positions
            preprocessed_cop_data = self.detrend_cop_data(cop_data)

            if save_cop:
                # Save intermediate results of COP computations
                save_as_json(preprocessed_cop_data, filepath, destination_folder="cop_results", name_extension="cop")
                return

            # Compute time features from COP displacement
            time_features = self.compute_time_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            # Compute frequency features from COP displacement
            frequency_features = self.compute_frequency_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            merged_features = {"filepath": filepath, "time_features": time_features, "frequency_features": frequency_features}

            # Save features in json format
            save_as_json(merged_features, filepath, destination_folder="feature_results", name_extension="features")

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def process_all_files(self, logger, balance_board=False):
        """Save features from all files."""

        if self.data is not None:
            for file in tqdm(self.data):
                self.save_features(file, balance_board)
        else:
            logger.critical("No files to process.")
            sys.exit()

    def set_pipeline_data(self, files):
        """Set the input data of the pipeline."""

        self.data = files
