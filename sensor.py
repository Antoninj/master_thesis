import btk
import json
# from utils import load_config


class SensorDataReader(object):
    """ Class to collect and format the raw sensor data values from the WBB and force plate acquisition files which are stored in the c3d file format (cf https://www.c3d.org/HTML/default.htm).

    The locally compiled and installed binaries of the biomechanical toolkit python wrapper (http://biomechanical-toolkit.github.io/docs/Wrapping/Python/index.html) are used to read and extract the data. """

    def __init__(self, filepath):
        self.config = self.load_config()
        self.acquisition_reader = btk.btkAcquisitionFileReader()
        self.set_reader_filename(filepath)

    def set_reader_filename(self, filepath):
        """ Function to change the current acquisition file used by the file reader """

        self.acquisition_reader.SetFilename(filepath)
        self.acquisition_reader.Update()

    @staticmethod
    def load_config():
        """ Function to load the configuration file """

        with open("config/sensor.json") as cfg:
            config = json.load(cfg)
        return config

    def get_raw_data(self, balance_board=False):
        """ Function to extract and aggregate raw sensor data of interest """

        reader = self.acquisition_reader
        acq = reader.GetOutput()
        if balance_board:
            labels = self.config["data_points_labels"]
            points = [acq.GetPoint(label)
                      for label in labels]
            values = [point.GetValues() for point in points]
        else:
            labels = self.config["analog_labels"]
            analogs = [acq.GetAnalog(label)
                       for label in labels]
            values = [analog.GetValues() for analog in analogs]

        return dict(zip(labels, values))

    def get_frequency(self, point=False):
        """ Function to extract analog/point frequencies """

        reader = self.acquisition_reader
        acq = reader.GetOutput()
        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()
