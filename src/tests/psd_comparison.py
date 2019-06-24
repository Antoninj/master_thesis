# Third-party module imports
import logging
import timeit
from argparse import ArgumentParser

from context import *
from matplotlib import pyplot as plt


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

    # scale factor to adjust WBB COP displacement
    scale_factor = config["preprocessing_parameters"]["scaling_factor"]

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="Preprocessing tests")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot frequency results")

    args = parser.parse_args()
    plot = args.plot

    # Create a sensor data reader object
    data_reader = HybridAcquisitionReader()

    # Create a data preprocessor object
    data_preprocessor = DataPreprocessor()

    ##################
    # Tests
    ##################

    logger.info("Processing Force Plate and WBB data.")
    logger.info("WBB Test file: {}".format(filepath_wbb))
    logger.info("FP Test file: {}".format(filepath_fp))

    fp_raw_data = data_reader.get_raw_data(filepath=filepath_fp)
    wbb_raw_data = data_reader.get_raw_data(filepath=filepath_wbb, balance_board=True)

    analog_freq = data_reader.get_frequency(filepath=filepath_fp)

    fp_preprocessed_cop_data = data_preprocessor.preprocess_raw_data(fp_raw_data)

    fp_cop_x = fp_preprocessed_cop_data["COP_x"]
    fp_cop_y = fp_preprocessed_cop_data["COP_y"]

    data_preprocessor.use_swarii = True
    data_preprocessor.window_size = 0.1
    wbb_preprocessed_cop_data = data_preprocessor.preprocess_raw_data(wbb_raw_data, True)

    # rescale wbb data
    for key in wbb_preprocessed_cop_data:
        wbb_preprocessed_cop_data[key] *= scale_factor

    wbb_cop_x = wbb_preprocessed_cop_data["COP_x"]
    wbb_cop_y = wbb_preprocessed_cop_data["COP_y"]

    # Frequency features computations
    logger.info("Computing frequency features.")

    fp_freq_features = FrequencyFeatures(fp_cop_x, fp_cop_y)
    wbb_freq_features = FrequencyFeatures(wbb_cop_x, wbb_cop_y)

    spectral_densities = ["ml_spectral_density_jackknife", "ap_spectral_density_jackknife",
                          "rd_spectral_density_jackknife"]

    fp_spectrums_and_frequencies = [getattr(fp_freq_features, spectral_density) for spectral_density in
                                    spectral_densities]
    fp_frequencies = [sd[0] for sd in fp_spectrums_and_frequencies]
    fp_spectrums = [sd[1] for sd in fp_spectrums_and_frequencies]
    fp_jackknifes = [sd[2] for sd in fp_spectrums_and_frequencies]

    wbb_spectrums_and_frequencies = [getattr(wbb_freq_features, spectral_density) for spectral_density in
                                     spectral_densities]
    wbb_frequencies = [sd[0] for sd in wbb_spectrums_and_frequencies]
    wbb_spectrums = [sd[1] for sd in wbb_spectrums_and_frequencies]
    wbb_jackknifes = [sd[2] for sd in wbb_spectrums_and_frequencies]

    if plot:
        plot_superposed_spectral_densities(fp_frequencies, fp_spectrums, fp_jackknifes, wbb_frequencies, wbb_spectrums,
                                           wbb_jackknifes)
        plt.savefig(
            "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/PSD_FP_WBB.png",
            bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()

    print('Execution time: {} seconds'.format(stop - start))
