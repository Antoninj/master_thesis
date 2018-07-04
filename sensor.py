import btk
import json
from utils import load_config


class SensorDataReader(object):
    """ Class to collect and format the raw sensor data values from the WBB and force plate acquisition files which are stored in the c3d file format (cd https://www.c3d.org/HTML/default.htm).

    The locally compiled binaries of the biomechanical toolkit python wrapper (http://biomechanical-toolkit.github.io/docs/Wrapping/Python/index.html) are used to read and extract the data. """

    def __init__(self, filepath):
        self.config = self.load_config()
        self.acquisition_reader = btk.btkAcquisitionFileReader()
        self.set_reader_filename(filepath)

    def set_reader_filename(self, filepath):
        self.acquisition_reader.SetFilename(filepath)
        self.acquisition_reader.Update()

    @staticmethod
    def load_config():
        with open("config/sensor.json") as cfg:
            config = json.load(cfg)
        return config

    def get_sensor_data(self, balance_board=False):
        reader = self.acquisition_reader
        acq = reader.GetOutput()

        if balance_board:
            data_points_labels = self.config["data_points_labels"]
            points = [acq.GetPoint(label)
                      for label in data_points_labels]
            values = [point.GetValues() for point in points]
        else:
            analog_labels = self.config["analog_labels"]

            analogs = [acq.GetAnalog(label)
                       for label in analog_labels]
            values = [analog.GetValues() for analog in analogs]

        return values

    def get_frequency(self, point=False):
        reader = self.acquisition_reader
        acq = reader.GetOutput()

        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()

    def format_sensor_data(self, data):
        pass
