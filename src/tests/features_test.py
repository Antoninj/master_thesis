# Third-party module imports
import logging
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from context import *

if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    # Setup logger
    setup_logging()
    logger = logging.getLogger("tests")

    # WBB data test file
    wbb_cop_data = config["test_files"]["wbb_cop_data"]

    # Command line argument parser to choose between frequency or time domain features
    parser = ArgumentParser(
        description="")
    parser.add_argument("-f", "--freq", action='store_true', help="Compute frequency features")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot results")

    args = parser.parse_args()
    freq = args.freq
    plot = args.plot

    logger.info("Testing feature computation modules.")
    logger.info("Test file: {}".format(wbb_cop_data))

    if not freq:
        # Time features computations
        logger.info("Computing time features.")

        time_features = TimeFeatures.from_file(wbb_cop_data)
        time_features.summary()

    else:
        # Frequency features computations
        logger.info("Computing frequency features.")
        freq_features = FrequencyFeatures.from_file(wbb_cop_data)

        (f, pxx) = freq_features.rd_spectral_density
        freq_features.summary()

        if plot:
            plt.semilogy(f, pxx)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [mm**2/Hz]')
            plt.show()
