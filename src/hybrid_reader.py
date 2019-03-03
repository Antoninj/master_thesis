# Third-party modules imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))

# Built-in modules imports
from btk import btkAcquisitionFileReader, btkAcquisition
from utils import load_config
from datetime import datetime

config = load_config()


class HybridAcquisitionReader(btkAcquisitionFileReader):
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
        super(HybridAcquisitionReader, self).__init__()

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
            values = [point.GetValues() for point in points]
        except RuntimeError:
            raise

        return dict(zip(labels, values))

    @staticmethod
    def compute_timestamps(time_dict):
        """ Reformat the acquisition timestamps from absolute dates to relative timestamps in seconds. """

        time_strings_lists = []
        for key, value in time_dict.items():
            flatten_values = time_dict[key].flatten()
            time_strings = ['{0:g}'.format(float(value)) for value in flatten_values]
            if key == "milisecond":
                for i in range(len(time_strings)):
                    if len(time_strings[i]) == 1:
                        time_strings[i] = "00" + time_strings[i]
                    if len(time_strings[i]) == 2:
                        time_strings[i] = "0" + time_strings[i]
            time_strings_lists.append(time_strings)

        date_strings = [" ".join(date) for date in list(zip(*time_strings_lists))]
        fmt = '%Y %m %d %H %M %S %f'
        datetimes = [datetime.strptime(string, fmt) for string in date_strings]

        timestamps_seconds = []
        duration = 0
        timestamps_seconds.append(duration)
        for i in range(len(datetimes) - 1):
            duration += (datetimes[i + 1] - datetimes[i]).total_seconds()
            timestamps_seconds.append(duration)

        return timestamps_seconds

    def get_raw_data(self, filepath, balance_board=False):
        """
        Extract and aggregate raw sensor data from the c3d acquisition file.

        The choice of data which is extracted can be modified through the configuration file.
        """

        self.set_reader_filename(filepath)
        acq = self.GetOutput()

        if balance_board:
            analog_labels = self.wbb_analog_labels
            data_points_labels = self.wbb_data_points_labels

            analog_data = self.get_analog_data(acq, analog_labels)
            relative_timestamps = self.compute_timestamps(analog_data)
            point_data = self.get_point_data(acq, data_points_labels)

            return [relative_timestamps, point_data]

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
