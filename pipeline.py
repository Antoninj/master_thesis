from sensor import SensorDataReader
from preprocess import DataPreprocessor
import json
from cop import *
import argparse
from utils import load_config, NumpyEncoder
import os


def compute_wbb_cop_positions(data_reader):

    raw_data = data_reader.get_raw_data(balance_board=True)
    cop_wbb_x = compute_cop_wbb_x(raw_data)
    cop_wbb_y = compute_cop_wbb_y(raw_data)
    cop_data = (cop_wbb_x, cop_wbb_y)

    return cop_data


def preprocess_wbb_cop_positions(data_preprocessor, cop_data):

    labels = ["COP x", "COP y"]
    preprocessed_data = [data_preprocessor.preprocess(data, 1000, True) for data in cop_data]

    return dict(zip(labels, preprocessed_data))


def compute_fp_cop_positions(data_reader):

    raw_data = data_reader.get_raw_data()
    cop_fp_x = compute_cop_fp_x(raw_data)
    cop_fp_y = compute_cop_fp_y(raw_data)
    cop_data = (cop_fp_x, cop_fp_y)

    return cop_data


def preprocess_fp_cop_positions(data_preprocessor, data_reader, cop_data):

    analog_freq = data_reader.get_frequency()
    labels = ["COP x", "COP y"]
    preprocessed_data = [data_preprocessor.preprocess(data, analog_freq) for data in cop_data]

    return dict(zip(labels, preprocessed_data))


def save_cop_positions(preprocessed_data, filepath):

    base_image_name = os.path.splitext(filepath)[0]
    filename = base_image_name + "_cop.json"
    with open(filename, 'w') as outfile:
            json.dump(preprocessed_data, outfile, cls=NumpyEncoder, sort_keys=True, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    filepath_wbb = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1.c3d"

    filepath_fp = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/Vicon/1.c3d"

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--wbb", action='store_true', help="Process WBB data")

    args = parser.parse_args()

    data_reader = SensorDataReader(filepath_wbb)
    data_preprocessor = DataPreprocessor()

    WBB = args.wbb
    if WBB:
        wbb_cop_positions = compute_wbb_cop_positions(data_reader)
        preprocessed_cop_positions = preprocess_wbb_cop_positions(data_preprocessor, wbb_cop_positions)

    else:
        data_reader.set_reader_filename(filepath_fp)
        fp_cop_positions = compute_fp_cop_positions(data_reader)
        preprocessed_cop_positions = preprocess_fp_cop_positions(data_preprocessor, data_reader, wbb_cop_positions)

    save_cop_positions(preprocessed_cop_positions, filepath_wbb)
