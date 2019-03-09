# Built-in modules imports
from hybrid_reader import HybridAcquisitionReader
from preprocessor import DataPreprocessor
from utils import save_as_json, plot_stabilograms
from tqdm import tqdm
import sys

# Third-party module imports
import logging

logger = logging.getLogger("pipeline")


class PreprocessingPipeline(HybridAcquisitionReader, DataPreprocessor):
    """
    Class that pipelines all the different data processing steps from acquisition file reading to feature extraction.
    """

    def __init__(self, acquisition_files=None):
        super(PreprocessingPipeline, self).__init__()
        self.acquisition_data = acquisition_files

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
            logger.info("Saving COP preprocessed data to file: {}".format(filepath))
            save_as_json(preprocessed_cop_data, filepath, folder_to_replace="BalanceBoard/Repro",
                         destination_folder="results/cop_data", name_extension="_cop.json")

            # Plot and save the stabilograms
            logger.info("Saving stabilograms plots to file: {}".format(filepath))
            plot_stabilograms(preprocessed_cop_data, device_name, self.acq_frequency, filepath=filepath)

        except Exception as err:
            logger.error(": {} \n Problem with file:{}".format(err, filepath), exc_info=True, stack_info=True)

    def preprocess_all_files(self, external_logger, balance_board=False):
        """Preprocess all c3d files."""

        if self.acquisition_data is not None:
            for acquisition_file in tqdm(self.acquisition_data):
                external_logger.debug("Preprocessing acquisition file: {}".format(acquisition_file))
                self.preprocess_acquisition_file(acquisition_file, balance_board)
        else:
            external_logger.critical("No files to preprocess.")
            sys.exit()

    def set_pipeline_acquisition_data(self, files):
        """Set the input acquisition data of the pipeline."""

        self.acquisition_data = files
