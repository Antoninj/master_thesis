import btk
from utils import load_config

config = load_config()


class SensorDataReader(object):
    """
    Class to collect and format the raw sensor data values from the WBB and force plate acquisition files which are stored in the c3d file format (cf https://www.c3d.org/HTML/default.htm).

    The locally compiled and installed binaries of the biomechanical toolkit python wrapper (http://biomechanical-toolkit.github.io/docs/Wrapping/Python/index.html) are used to read and extract the data.
    """

    # Constants
    data_labels = config["sensor_labels"]["data_points_labels"]
    analog_labels = config["sensor_labels"]["analog_labels"]

    def __init__(self):
        self.acquisition_reader = btk.btkAcquisitionFileReader()

    def set_reader_filename(self, filepath):
        """ Function to change the current acquisition file used by the file reader """

        self.acquisition_reader.SetFilename(filepath)
        self.acquisition_reader.Update()

    def get_raw_data(self, balance_board=False):
        """ Function to extract and aggregate raw sensor data of interest """

        acq = self.acquisition_reader.GetOutput()
        if balance_board:
            labels = self.data_labels
            points = [acq.GetPoint(label)
                      for label in labels]
            values = [point.GetValues() for point in points]
        else:
            labels = self.analog_labels
            analogs = [acq.GetAnalog(label)
                       for label in labels]
            values = [analog.GetValues() for analog in analogs]

        return dict(zip(labels, values))

    def get_frequency(self, point=False):
        """ Function to extract analog/point frequencies """

        acq = self.acquisition_reader.GetOutput()
        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()
