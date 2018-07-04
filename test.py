from sensor import SensorDataReader
from preprocess import DataPreprocessor
import json
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Load configuration file
    with open("config/config.json") as cfg:
        config = json.load(cfg)

    filepath_wb = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1.c3d"

    filepath_fp = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/Vicon/1.c3d"

    data_reader = SensorDataReader(filepath_fp)

    sensor_data = data_reader.get_sensor_data(balance_board=False)
    print(sensor_data[0])

    analog_freq = data_reader.get_frequency()
    point_freq = data_reader.get_frequency(point=True)

    print(analog_freq)

    data_preprocessor = DataPreprocessor()
    filtered_data = data_preprocessor.apply_filtering(
        sensor_data[0], analog_freq)

    print(filtered_data)

    filtered_detrended = data_preprocessor.apply_detrending(filtered_data)

    print(filtered_detrended)

    plt.figure()
    plt.plot(sensor_data[0][:, 0])
    plt.plot(filtered_detrended)
    plt.show()
