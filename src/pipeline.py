# Built-in modules imports
from hybrid_reader import HybridAcquisitionReader
from preprocessor import DataPreprocessor
from processor import DataProcessor
from utils import save_as_json, plot_stabilograms
from tqdm import tqdm
import sys

# Third-party module imports
import logging

logger = logging.getLogger("pipeline")


class DataPipeline(HybridAcquisitionReader, DataPreprocessor, DataProcessor):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction.
    """

    def __init__(self, acquisition_files=None, preprocessed_files=None):
        super(DataPipeline, self).__init__()
        self.acquisition_data = acquisition_files
        self.cop_data = preprocessed_files

    def preprocess_acquisition_file(self, filepath, balance_board=False):
        """
        Pipeline the raw acquisition file reading, COP computations and preprocessing steps and save the results to a json file."""

        try:
            # Get the raw data
            raw_data = self.get_raw_data(filepath, balance_board)

            # Preprocess the raw data
            preprocessed_cop_data = self.preprocess_raw_data(raw_data, balance_board)

            if balance_board:
                device_name = "Wii Balance Board"
            else:
                device_name = "Force plate"

            # Save results of COP signal computations and preprocessing
            save_as_json(preprocessed_cop_data, filepath, folder_to_replace="BalanceBoard/Repro", destination_folder="results/cop_data", name_extension="cop.json")

            # Plot and save the stabilograms
            plot_stabilograms(preprocessed_cop_data, device_name, self.acq_frequency, filepath=filepath)

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def process_cop_data_file(self, filepath):
        """
        Pipeline the COP data processing, i.e the time and frequency feature extraction steps, and save the results to a json file."""

        try:
            # Compute time features from COP displacement
            time_features = self.compute_time_features(filepath)

            # Compute frequency features from COP displacement
            frequency_features = self.compute_frequency_features(filepath)

            file_info = self.parse_filepath(filepath)

            processed_data = {**file_info, "time_features": time_features, "frequency_features": frequency_features}

            # Save features computations in json format
            save_as_json(processed_data, filepath, folder_to_replace="cop_data", destination_folder="feature_data", name_extension="features.json")

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def preprocess_all_files(self, logger, balance_board=False):
        """Preprocess all c3d files."""

        if self.acquisition_data is not None:
            for acquisition_file in tqdm(self.acquisition_data):
                logger.debug("Preprocessing acquisition file: {}".format(acquisition_file))
                self.preprocess_acquisition_file(acquisition_file, balance_board)
        else:
            logger.critical("No files to preprocess.")
            sys.exit()

    def process_all_files(self, logger):
        """Compute features from all preprocessed files."""

        if self.cop_data is not None:
            for cop_data_file in tqdm(self.cop_data):
                logger.debug("Processing COP data file: {}".format(cop_data_file))
                self.process_cop_data_file(cop_data_file)
        else:
            logger.critical("No files to process.")
            sys.exit()

    def set_pipeline_acquisition_data(self, files):
        """Set the input acquisition data of the pipeline."""

        self.acquisition_data = files

    def set_pipeline_cop_data(self, files):
        """Set the input cop data of the pipeline."""

        self.cop_data = files

    @staticmethod
    def parse_filepath(file):
        """Parse the filepath to retrieve information that allows to identify the acquisition"""

        FP = "FP"
        WBB = "BB"
        keys = ["device", "subject", "trial", "balance board"]
        trial_info = dict.fromkeys(keys)
        trial_info["device"] = FP if FP in file else WBB

        pre_subject_substring = "Repro"
        pre_subject_substring_index = file.find(pre_subject_substring)
        trial_info["subject"] = file[pre_subject_substring_index + len(pre_subject_substring)]

        pre_balance_board_substring = trial_info["device"] + "/"
        pre_balance_board_substring_index = file.find(pre_balance_board_substring)
        trial_info["balance board"] = file[pre_balance_board_substring_index + len(pre_balance_board_substring)]

        pre_trial_substring = pre_balance_board_substring + trial_info["balance board"] + "_"
        pre_trial_substring_index = file.find(pre_trial_substring)
        trial_info["trial"] = file[pre_trial_substring_index + len(pre_trial_substring)]

        return trial_info
