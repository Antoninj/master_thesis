# Third-party modules imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))

# Built-in modules imports
from btk import btkAcquisitionFileReader, btkAcquisition
from utils import load_config

config = load_config()


class SensorDataReader(btkAcquisitionFileReader):
    """
    Class used to extract the raw sensor data values from the WBB and force plate acquisition files which are stored in the c3d file format.

    The locally compiled and installed binaries of the biomechanical toolkit python wrapper are used to read
    and extract the data.

    Notes
    -----
    Biomechanical toolkit python wrapper starting guide: http://biomechanical-toolkit.github.io/docs/Wrapping/Python/index.html
    C3D format documentation: https://www.c3d.org/HTML/default.htm

    """

    wbb_data_points_labels = config["wbb_labels"]["data_points_labels"]
    wbb_analog_labels = config["wbb_labels"]["analog_labels"]
    force_plate_analog_labels = config["force_plate_labels"]["analog_labels"]

    def __init__(self):
        super(SensorDataReader, self).__init__()

    def set_reader_filename(self, filepath):
        """Change the acquisition file currently used by the file reader."""

        self.SetFilename(filepath)
        self.Update()

    @staticmethod
    def get_point_data(acquisition, labels):
        try:
            points = [acquisition.GetPoint(label) for label in labels]
            values = [point.GetValues() for point in points]
        except RuntimeError:
            raise

        return dict(zip(labels, values))

    @staticmethod
    def get_analog_data(acquisition, labels):
        try:
            points = [acquisition.GetAnalog(label) for label in labels]
            """
            print("Gain: ", points[0].GetGain())
            print("Scale: ", points[0].GetScale())
            print("Offset: ", points[0].GetOffset())
            print("Unit: ", points[0].GetUnit())
            """
            values = [point.GetValues() for point in points]
        except RuntimeError:
            raise

        return dict(zip(labels, values))

    def get_raw_data(self, filepath, balance_board=False):
        """
        Extract and aggregate raw sensor data from the c3d acquisition file.

        The data that is extracted can be modified through the configuration file.
        """

        self.set_reader_filename(filepath)
        acq = self.GetOutput()

        """
        resolution = acq.GetAnalogResolution()
        print("Analog resolution: {}".format(resolution))
        acq.SetAnalogResolution(btkAcquisition.Bit16)
        resolution = acq.GetAnalogResolution()
        print("New analog resolution: {}".format(resolution))
        """

        if balance_board:
            analog_labels = self.wbb_analog_labels
            data_points_labels = self.wbb_data_points_labels

            analog_data = self.get_analog_data(acq, analog_labels)
            point_data = self.get_point_data(acq, data_points_labels)

            return [analog_data, point_data]

        else:
            analog_labels = self.force_plate_analog_labels
            analog_data = self.get_analog_data(acq, analog_labels)

            return analog_data

    def get_frequency(self, filepath, point=False):
        """Extract analog/point frequencies."""

        self.set_reader_filename(filepath)
        acq = self.GetOutput()

        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()
