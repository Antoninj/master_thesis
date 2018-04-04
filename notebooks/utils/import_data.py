import pandas as pd
import os


def get_filenames(data_path):
    return ["".join([data_path, "/", filename]) for filename in os.listdir(data_path) if
            not(filename.startswith(".")) and filename.endswith("csv") ]


def read_data(data_path):
    filenames = get_filenames(data_path)
    dataframes = [pd.read_csv(filename, sep=",") for filename in filenames]
    return dataframes
