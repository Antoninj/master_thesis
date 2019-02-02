# Third-party module imports
import logging
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from context import *

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
    wbb_cop_data = config["test_files"]["wbb_cop_data"]

    # FP data test file
    fp_cop_data = config["test_files"]["fp_cop_data"]

    # Command line argument parser to choose between frequency or time domain features
    parser = ArgumentParser(
        description="")
    parser.add_argument("-w", "--wbb", action='store_true', help="Compute WBB features")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot frequency results")

    args = parser.parse_args()
    WBB = args.wbb
    plot = args.plot

    ##################
    # Tests
    ##################

    if WBB:
        data_file = wbb_cop_data
    else:
        data_file = fp_cop_data

    logger.info("Testing feature computation modules.")
    logger.info("Test file: {}".format(data_file))

    # Time features computations 
    logger.info("Computing time features.")
    time_features = TimeFeatures.from_file(data_file)
    time_features.summary()

    # Frequency features computations
    logger.info("Computing frequency features.")
    freq_features = FrequencyFeatures.from_file(data_file)

    (f, pxx) = freq_features.rd_spectral_density
    freq_features.summary()

    print(f)

    if plot:
        plt.plot(f, pxx)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [mm**2/Hz]')
        plt.show()
