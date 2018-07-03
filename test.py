from sensor import SensorDataReader
import json

if __name__ == "__main__":

    # Load configuration file
    with open("config/config.json") as cfg:
        config = json.load(cfg)

    filepath = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1.c3d"

    data_reader = SensorDataReader(filepath)

    wbb_sensor_data = data_reader.get_sensor_data(balance_board=True)
    print(wbb_sensor_data)

    analog_freq = data_reader.get_frequency()
    point_freq = data_reader.get_frequency(point=True)
