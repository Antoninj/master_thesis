# Third-party module imports
import logging
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from context import *


if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    # Setup logger
    logger = logging.getLogger(__name__)
    setup_logging()

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

    logger.info("This is a test")

    if WBB:
        raw_data = data_reader.get_raw_data(filepath=filepath_wbb, balance_board=True)
        analog_freq = data_reader.get_frequency(filepath=filepath_wbb)

        #print(raw_data, analog_freq)

        cop_wbb_x = compute_cop_wbb_x(raw_data)
        cop_wbb_y = compute_cop_wbb_y(raw_data)

        print("WBB COP x: {} \nWBB COP y: {}".format(cop_wbb_x, cop_wbb_y))

        preprocessed_data = data_preprocessor.preprocess(cop_wbb_x, 1000, True)

        if plot:
            plt.figure()
            plt.plot(cop_wbb_x)
            plt.plot(preprocessed_data)
            plt.show()

    else:
        raw_data = data_reader.get_raw_data(filepath=filepath_fp)
        analog_freq = data_reader.get_frequency(filepath=filepath_fp)

        #print(raw_data["Fz1"], analog_freq)
        print(len(raw_data["Fz1"]))
        cop_fp_x = compute_cop_fp_x(raw_data)
        cop_fp_y = compute_cop_fp_y(raw_data)

        print("FP COP x: {} \nFP COP y: {}".format(cop_fp_x, cop_fp_y))

        preprocessed_data = data_preprocessor.preprocess(cop_fp_x, analog_freq)

        if plot:
            plt.figure()
            plt.plot(cop_fp_x)
            plt.plot(preprocessed_data)
            plt.show()
