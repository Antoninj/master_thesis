# Third-party module imports
import logging
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np
from context import *


def main():
    ##################
    # Boilerplate code
    ##################

    # Load configuration file
    config = load_config()

    # Setup logger
    setup_logging()
    logger = logging.getLogger("tests")

    # WBB data test file
    filepath_wbb = config["test_files"]["wbb_raw_data"]

    # Force plate data test file
    filepath_fp = config["test_files"]["fp_raw_data"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="Preprocessing tests")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    parser.add_argument("-f", "--fp", action='store_true', help="Process Force plate data")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot data")
    parser.add_argument("-d", "--debug", action='store_true', help="Enable debug mode")


    args = parser.parse_args()
    WBB = args.wbb
    plot = args.plot
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    # Create a sensor data reader object
    data_reader = HybridAcquisitionReader()

    # Create a data preprocessor reader object
    data_preprocessor = DataPreprocessor()

    ##################
    # Tests
    ##################

    logger.info("Testing preprocessing module")

    if WBB:
        device_name = "Wii Balance Board"
        logger.info("Processing Wii Balance Board data.")
        logger.info("Test file: {}".format(filepath_wbb))

        raw_data = data_reader.get_raw_data(filepath=filepath_wbb, balance_board=True)
        analog_freq = data_reader.get_frequency(filepath=filepath_wbb)

        logger.debug("Analog data: {} \nAnalog Frequency: {}".format(raw_data[0], analog_freq))
        logger.debug("Point data :{}".format(raw_data[1]))

        logger.debug("Data size: {}".format(len(raw_data[1]["Accelerometer"])))

        preprocessed_cop_data = data_preprocessor.preprocess_raw_data(raw_data, True)

        cop_x = preprocessed_cop_data["COP_x"]
        cop_y = preprocessed_cop_data["COP_y"]

        logger.debug("WBB COP x: {} \n WBB COP y: {}".format(cop_x, cop_y))

    else:
        device_name = "Force plate"
        logger.info("Processing Force Plate data.")
        logger.info("Test file: {}".format(filepath_fp))

        raw_data = data_reader.get_raw_data(filepath=filepath_fp)
        analog_freq = data_reader.get_frequency(filepath=filepath_fp)

        logger.debug("Raw data: {} \nAnalog Frequency: {}".format(raw_data, analog_freq))

        logger.debug("Fz1 size: {}".format(len(raw_data["Fz1"])))

        preprocessed_cop_data = data_preprocessor.preprocess_raw_data(raw_data)

        cop_x = preprocessed_cop_data["COP_x"]
        cop_y = preprocessed_cop_data["COP_y"]

        logger.debug("FP COP x: {} \n FP COP y: {}".format(cop_x, cop_y))

    if plot:
        acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
        plot_stabilograms(preprocessed_cop_data, device_name, acq_frequency)
        plt.show()


if __name__ == "__main__":

    main()
