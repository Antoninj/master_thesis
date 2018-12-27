# Built-in modules imports
from sensor import SensorDataReader
from preprocess import DataPreprocessor
from process import DataProcessor
from utils import save_as_json, plot_stabilograms
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

    def process_file(self, filepath, balance_board=False, save_cop=False):
        """
        Pipeline the preprocessing, COP computations and feature extraction steps and save the results to a json file."""

        try:
            # Get the raw data
            raw_data = self.get_raw_data(filepath, balance_board)

            # Preprocess the raw data
            preprocessed_cop_data = self.preprocess_raw_data(raw_data, balance_board)

            file_info = self.parse_filepath(filepath)

            if save_cop:
                if balance_board:
                    device_name = "Wii Balance Board"
                else:
                    device_name = "Force plate"

                # Save intermediate results of COP computations
                save_as_json(preprocessed_cop_data, filepath, destination_folder="cop_data", name_extension="cop.json")
                plot_stabilograms(preprocessed_cop_data, device_name, self.acq_frequency, filepath=filepath)

            # Compute time features from COP displacement
            time_features = self.compute_time_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            # Compute frequency features from COP displacement
            frequency_features = self.compute_frequency_features(preprocessed_cop_data["COP_x"], preprocessed_cop_data["COP_y"])

            processed_data = {**file_info, "time_features": time_features, "frequency_features": frequency_features}

            # Save features in json format
            save_as_json(processed_data, filepath, destination_folder="feature_data", name_extension="features.json")

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def process_all_files(self, logger, balance_board=False, save_cop=False):
        """Save features from all files."""

        if self.data is not None:
            for file in tqdm(self.data):
                logger.info("Processing file: {}".format(file))
                self.process_file(file, balance_board, save_cop)
        else:
            logger.critical("No files to process.")
            sys.exit()

    def set_pipeline_data(self, files):
        """Set the input data of the pipeline."""

        self.data = files

    @staticmethod
    def parse_filepath(file):
        FP = "FP"
        WBB = "BB"
        keys = ["device", "subject", "trial", "balance board"]
        trial_info = dict.fromkeys(keys)
        trial_info["device"] = FP if FP in file else WBB

        pre_subject_substring = "Repro/Repro"
        pre_subject_substring_index = file.find(pre_subject_substring)
        trial_info["subject"] = file[pre_subject_substring_index + len(pre_subject_substring)]

        pre_balance_board_substring = trial_info["device"] + "/"
        pre_balance_board_substring_index = file.find(pre_balance_board_substring)
        trial_info["balance board"] = file[pre_balance_board_substring_index + len(pre_balance_board_substring)]

        pre_trial_substring = trial_info["device"] + "/" + trial_info["balance board"] + "_"
        pre_trial_substring_index = file.find(pre_trial_substring)
        trial_info["trial"] = file[pre_trial_substring_index + len(pre_trial_substring)]

        return trial_info
