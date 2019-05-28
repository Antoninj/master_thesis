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

    # Command line argument parser to choose between wbb or force plate data
    parser = ArgumentParser(
        description="Preprocessing tests")
    parser.add_argument("-w", "--wbb", action='store_true', help="Process WBB data")
    parser.add_argument("-f", "--fp", action='store_true', help="Process Force plate data")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot data")
    parser.add_argument("-d", "--debug", action='store_true', help="Enable debug mode")


    args = parser.parse_args()
    WBB = args.wbb
    FP = args.fp
    plot = args.plot
    debug = args.debug

    if debug:
        logger.setLevel("DEBUG")

    # Create a sensor data reader object
    data_reader = HybridAcquisitionReader()

    # Create a data preprocessor object
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
        acquisition_duration = raw_data[0][-1]
        acquisition_sample_size = len(raw_data[0])
        wbb_avg_freq = acquisition_sample_size / acquisition_duration
        logger.debug("Acquisition duration: {} s".format(acquisition_duration))
        logger.debug("Time stamps: {} \nWBB average sampling frequency: {} Hz".format(raw_data[0], wbb_avg_freq))
        logger.debug("Point data :{}".format(raw_data[1]))

        logger.debug("Raw data size: {}".format(len(raw_data[1]["Accelerometer"])))

        data_preprocessor.use_swarii = False
        preprocessed_cop_data_no_swarii = data_preprocessor.preprocess_raw_data(raw_data, True)

        data_preprocessor.use_swarii = True
        preprocessed_cop_data_swarii = data_preprocessor.preprocess_raw_data(raw_data, True)

        logger.debug("Preprocessed data size: {}".format(len(preprocessed_cop_data_no_swarii["COP_x"])))

        cop_x = preprocessed_cop_data_no_swarii["COP_x"]
        cop_y = preprocessed_cop_data_no_swarii["COP_y"]

        logger.debug("WBB COP x: {} \n WBB COP y: {}".format(cop_x, cop_y))

        if plot:
            acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
            plot_swarii_comparison_stabilograms(preprocessed_cop_data_no_swarii, preprocessed_cop_data_swarii,
                                                device_name, acq_frequency,
                                                "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/stabilogram_statokinesigram_WBB.png")
            plt.show()

    elif FP:
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

        logger.debug("FP COP x: {} \n FP COP y: {} \n".format(cop_x, cop_y))

        if plot:
            acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
            plot_stabilograms(preprocessed_cop_data, device_name, acq_frequency)
            plt.savefig(
                "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/stabilogram_statokinesigram_FP.png",
                bbox_inches='tight')
            plt.show()

    else:
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
        wbb_preprocessed_cop_data = data_preprocessor.preprocess_raw_data(wbb_raw_data, True)

        wbb_cop_x = wbb_preprocessed_cop_data["COP_x"]
        wbb_cop_y = wbb_preprocessed_cop_data["COP_y"]

        if plot:
            acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
            plot_superposed_stabilograms(fp_preprocessed_cop_data, wbb_preprocessed_cop_data, acq_frequency)
            plt.savefig(
                "/Users/Antonin/Documents/VUB/semester 4/thesis/paper/images/chapter 4/WBB_FP_superposed_stabilogram_statokinesigram.png",
                bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()

    print('Execution time: {} seconds'.format(stop - start))
