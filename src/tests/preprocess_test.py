# Third-party module imports
import logging
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np

from context import *


def plot_stabilograms(cop_x, cop_y, cop_rd, device_name):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    index = [i / 1000 for i in range(len(cop_x))]
    fig.suptitle("{} plots".format(device_name), fontsize=16)

    axs[0][0].plot(index, cop_x)
    axs[0][0].set_xlabel('Time (seconds)')
    axs[0][0].set_ylabel('ML distance (mm)')
    axs[0][1].plot(index, cop_y)
    axs[0][1].set_xlabel('Time (seconds)')
    axs[0][1].set_ylabel('AP distance (mm)')
    axs[1][0].plot(index, cop_rd)
    axs[1][0].set_xlabel('Time (seconds)')
    axs[1][0].set_ylabel('Resultant distance (mm)')
    axs[1][1].plot(cop_x, cop_y)
    axs[1][1].set_xlabel('ML distance (mm)')
    axs[1][1].set_ylabel('AP distance (mm)')


def compute_rd(cop_x, cop_y):
    return np.array([np.sqrt(x**2 + y**2) for x, y in zip(cop_x, cop_y)])


if __name__ == "__main__":

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
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot data")
    args = parser.parse_args()
    WBB = args.wbb
    plot = args.plot

    # Create a sensor data reader object
    data_reader = SensorDataReader()

    # Create a data preprocessor reader object
    data_preprocessor = DataPreprocessor()

    # Create a data preprocessor reader object
    data_processor = DataProcessor()

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
        cop_rd = compute_rd(cop_x, cop_y)
        plot_stabilograms(cop_x, cop_y, cop_rd, device_name)
        plt.show()
