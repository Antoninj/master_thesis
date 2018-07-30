# Third-party modules imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))

# Built-in modules imports
from btk import btkAcquisitionFileReader
from utils import load_config

config = load_config()


class SensorDataReader(btkAcquisitionFileReader):
    """
    Class to collect and format the raw sensor data values from the WBB and force plate acquisition files which are stored in the c3d file format (cf https://www.c3d.org/HTML/default.htm).

    The locally compiled and installed binaries of the biomechanical toolkit python wrapper (http://biomechanical-toolkit.github.io/docs/Wrapping/Python/index.html) are used to read
    and extract the data.
    """

    data_labels = config["sensor_labels"]["data_points_labels"]
    analog_labels = config["sensor_labels"]["analog_labels"]

    def __init__(self):
        super(SensorDataReader, self).__init__()

    def set_reader_filename(self, filepath):
        """Change the current acquisition file used by the file reader."""

        self.SetFilename(filepath)
        self.Update()

    def get_raw_data(self, filepath, balance_board=False):
        """Extract and aggregate raw sensor data of interest."""

        self.set_reader_filename(filepath)
        acq = self.GetOutput()

        if balance_board:
            labels = self.data_labels
            try:
                points = [acq.GetPoint(label)
                          for label in labels]
                values = [point.GetValues() for point in points]
            except RuntimeError:
                raise

        else:
            labels = self.analog_labels
            try:
                analogs = [acq.GetAnalog(label)
                           for label in labels]
                values = [analog.GetValues() for analog in analogs]
            except RuntimeError:
                raise

        return dict(zip(labels, values))

    def get_frequency(self, filepath, point=False):
        """Extract analog/point frequencies."""

        self.set_reader_filename(filepath)
        acq = self.GetOutput()

        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()
