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
        device = "WBB"
    else:
        data_file = fp_cop_data
        device = "FP"

    logger.info("Testing feature computation modules.")
    logger.info("Test file: {}".format(data_file))

    # Time features computations 
    logger.info("Computing time features.")
    time_features = TimeFeatures.from_file(data_file)
    time_features.summary()

    # Frequency features computations
    logger.info("Computing frequency features.")
    freq_features = FrequencyFeatures.from_file(data_file)

    spectral_densities = ["ml_spectral_density", "ap_spectral_density", "rd_spectral_density"]
    spectrums_and_frequencies = [getattr(freq_features, spectral_density) for spectral_density in spectral_densities]
    frequencies = [sd[0] for sd in spectrums_and_frequencies]
    spectrums = [sd[1] for sd in spectrums_and_frequencies]

    freq_features.summary()

    if plot:
        plot_spectral_densities(frequencies, spectrums)
        plt.savefig(
            "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/PSD_{}.png".format(device),
            bbox_inches='tight')

        plt.show()
