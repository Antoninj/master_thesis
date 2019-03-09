# Built-in modules imports
from hybrid_reader import HybridAcquisitionReader
from processor import DataProcessor
from utils import save_as_json, plot_stabilograms, plot_spectral_densities
from tqdm import tqdm
import sys

# Third-party module imports
import logging

logger = logging.getLogger("pipeline")


class FeatureExtractionPipeline(HybridAcquisitionReader, DataProcessor):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction.
    """

    def __init__(self, preprocessed_files=None):
        super(FeatureExtractionPipeline, self).__init__()
        self.cop_data = preprocessed_files

    def process_cop_data_file(self, filepath):
        """
        Pipeline the COP data processing, i.e the time and frequency feature extraction steps, and save the results to a json file."""

        try:
            # Compute time features from COP displacement
            time_features = self.compute_time_features(filepath)

            # Compute frequency features from COP displacement
            frequency_domain_features = self.compute_frequency_features(filepath)
            frequency_features = frequency_domain_features.frequency_features

            file_info = self.parse_filepath(filepath)

            processed_data = {**file_info, "time_features": time_features, "frequency_features": frequency_features}

            # Save features computations in json format
            logger.info("Saving time and frequency features to file: {}".format(filepath))
            save_as_json(processed_data, filepath, folder_to_replace="cop_data",
                         destination_folder="feature_data", name_extension="_features.json")

            # Plot and save the spectral densities
            spectral_densities = ["ap_spectral_density", "ml_spectral_density", "rd_spectral_density"]
            spectrums_and_frequencies = [getattr(frequency_domain_features, spectral_density) for spectral_density in
                                         spectral_densities]
            frequencies = [sd[0] for sd in spectrums_and_frequencies]
            spectrums = [sd[1] for sd in spectrums_and_frequencies]

            logger.info("Saving spectral density plot to file: {}".format(filepath))
            plot_spectral_densities(frequencies, spectrums, filepath=filepath)

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def process_all_files(self, external_logger):
        """Compute features from all preprocessed files."""

        if self.cop_data is not None:
            for cop_data_file in tqdm(self.cop_data):
                external_logger.debug("Processing COP data file: {}".format(cop_data_file))
                self.process_cop_data_file(cop_data_file)
        else:
            external_logger.critical("No files to process.")
            sys.exit()

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
