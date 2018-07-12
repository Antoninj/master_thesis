from sensor import SensorDataReader
from preprocess import DataPreprocessor
from cop import *
from matplotlib import pyplot as plt
import argparse
config = load_config()

if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    # WBB data test file
    filepath_wbb = config["test_files"]["wbb_raw_data"]

    # Force plate data test file
    filepath_fp = config["test_files"]["fp_raw_data"]

    # Command line argument parser to choose between wbb or force plate data
    parser = argparse.ArgumentParser(
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

    if WBB:
        data_reader.set_reader_filename(filepath_wbb)
        raw_data = data_reader.get_raw_data(balance_board=True)
        analog_freq = data_reader.get_frequency()

        #print(raw_data, analog_freq)

        cop_wbb_x = compute_cop_wbb_x(raw_data)
        cop_wbb_y = compute_cop_wbb_y(raw_data)

        print(len(cop_wbb_x))

        print("WBB COP x: {} \nWBB COP y: {}".format(cop_wbb_x, cop_wbb_y))

        preprocessed_data = data_preprocessor.preprocess(cop_wbb_x, 1000, True)

        print(len(preprocessed_data))

        if plot:
            plt.figure()
            # plt.plot(cop_wbb_x)
            plt.plot(preprocessed_data)
            plt.show()

    else:
        data_reader.set_reader_filename(filepath_fp)
        raw_data = data_reader.get_raw_data()
        analog_freq = data_reader.get_frequency()

        print(raw_data["Fz1"], analog_freq)

        cop_fp_x = compute_cop_fp_x(raw_data)
        cop_fp_y = compute_cop_fp_y(raw_data)

        print(len(cop_fp_x))
        print("FP COP x: {} \nFP COP y: {}".format(cop_fp_x, cop_fp_y))

        #preprocessed_data = data_preprocessor.preprocess(cop_fp_x, analog_freq)

        preprocessed_cop_fp_x = data_preprocessor.apply_filtering(cop_fp_x, analog_freq)
        print(preprocessed_cop_fp_x)
        #filtered_detrended = data_preprocessor.apply_detrending(preprocessed_cop_fp_x)

        if plot:
            plt.figure()
            plt.plot(cop_fp_x)
            # plt.plot(preprocessed_cop_fp_x)
            plt.show()
