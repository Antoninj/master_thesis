# Third-party module imports
import logging
import timeit
from argparse import ArgumentParser

from context import *
from matplotlib import pyplot as plt
import numpy as np


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

    # scale factor to adjust WBB COP displacement
    scale_factor = config["preprocessing_parameters"]["scaling_factor"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="SWARII time window stufy")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot data")
    parser.add_argument("-d", "--debug", action='store_true', help="Enable debug mode")

    args = parser.parse_args()
    plot = args.plot
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    swarii_time_windows = [0.1, 0.5, 1, 1.5, 2, 5]

    # Create a sensor data reader object
    data_reader = HybridAcquisitionReader()

    # Create a data preprocessor object
    data_preprocessors = [DataPreprocessor() for i in range(len(swarii_time_windows))]

    for (dp, time_window) in zip(data_preprocessors, swarii_time_windows):
        logger.info("Swarii time window: {}".format(time_window))
        dp.use_swarii = True
        dp.window_size = time_window

    # Get raw data
    raw_data = data_reader.get_raw_data(filepath=filepath_wbb, balance_board=True)

    preprocessed_cop_data_swariis = [dp.preprocess_raw_data(raw_data, True) for dp in data_preprocessors]
    for preprocessed_cop_data in preprocessed_cop_data_swariis:
        for key in preprocessed_cop_data:
            preprocessed_cop_data[key] *= scale_factor

    if plot:
        acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
        plot_multiple_swarii(preprocessed_cop_data_swariis, swarii_time_windows, acq_frequency,
                             "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/SWARII_time_window_impact.png")
        plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()

    print('Execution time: {} seconds'.format(stop - start))
