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
    filepath_wbb = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1.c3d"

    # Force plate data test file
    filepath_fp = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/Vicon/1.c3d"

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--wbb", action='store_true', help="Process WBB data")

    args = parser.parse_args()

    WBB = args.wbb
    if WBB:
        data_reader = SensorDataReader(filepath_wbb)
        raw_data = data_reader.get_raw_data(balance_board=True)
        analog_freq = data_reader.get_frequency()

        #print(raw_data, analog_freq)

        cop_wbb_x = compute_cop_wbb_x(raw_data)
        cop_wbb_y = compute_cop_wbb_y(raw_data)

        print("WBB COP x: {} \nWBB COP y: {}".format(cop_wbb_x, cop_wbb_y))

        data_preprocessor = DataPreprocessor()
        preprocessed_data = data_preprocessor.preprocess(cop_wbb_x, 1000, True)

        #preprocessed_cop_wbb_x = data_preprocessor.apply_filtering(cop_wbb_x, analog_freq)
        #filtered_detrended = data_preprocessor.apply_detrending(preprocessed_cop_wbb_x)

        plot_figure = True

        if plot_figure:
            plt.figure()
            plt.plot(cop_wbb_x)
            plt.plot(preprocessed_data)
            plt.show()

    else:
        data_reader = SensorDataReader(filepath_fp)
        raw_data = data_reader.get_raw_data()
        analog_freq = data_reader.get_frequency()

        #print(raw_data["Fz1"], analog_freq)

        cop_fp_x = compute_cop_fp_x(raw_data)
        cop_fp_y = compute_cop_fp_y(raw_data)

        print("FP COP x: {} \nFP COP y: {}".format(cop_fp_x, cop_fp_y))

        data_preprocessor = DataPreprocessor()
        #preprocessed_data = data_preprocessor.preprocess(cop_fp_x, analog_freq)

        preprocessed_cop_fp_x = data_preprocessor.apply_filtering(cop_fp_x, analog_freq)
        print(preprocessed_cop_fp_x)
        #filtered_detrended = data_preprocessor.apply_detrending(preprocessed_cop_fp_x)

        plot_figure = False

        if plot_figure:
            plt.figure()
            plt.plot(cop_fp_x)
            plt.plot(preprocessed_data)
            plt.show()
