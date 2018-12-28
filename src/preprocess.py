# Third-party module imports
import scipy.signal
from datetime import datetime

# Built-in modules imports
from utils import load_config
from resampling import SWARII
import pandas as pd
import numpy as np
import warnings
import logging

config = load_config()

# Set numpy error level to warning
np.seterr(all='warn')

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('error')

logging.captureWarnings(True)


class DataPreprocessor(SWARII):
    """
    Class that handles the preprocessing related tasks of the acquisition signal using the Scipy open source library and an open source implementation of the SWARII algorithm.

    More specifically, it relies on the implementations found in the latest release of the scipy signal processing package scipy.signal.

    References
    ----------
    .. [1] Scipy: https://scipy.org
    .. [2] Scipy.signal documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/signal.html
    .. [3] SWARII implementation : https://reine.cmla.ens-cachan.fr/j.audiffren/SWARII
    """

    up = config["preprocessing_parameters"]["upsampling_factor"]
    down = config["preprocessing_parameters"]["downsampling_factor"]
    order = config["preprocessing_parameters"]["filter_order"]
    fc = config["preprocessing_parameters"]["cutoff_frequency"]
    detrending_type = config["preprocessing_parameters"]["detrending_type"]
    low_thresh = config["preprocessing_parameters"]["lower_threshold"]
    up_thresh = config["preprocessing_parameters"]["upper_threshold"]
    time_shift = config["preprocessing_parameters"]["time_shift"]
    swarii_window = config["preprocessing_parameters"]["swarii_window_size"]
    acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
    use_swarii = config["preprocessing_parameters"]["apply_swarii"]

    def __init__(self):
        super(DataPreprocessor, self).__init__(window_size=self.swarii_window, desired_frequency=self.acq_frequency)

    def apply_swarii(self, input_signal, timestamps):
        """Apply the SWARII to resample a given signal."""

        resampled_time, resampled_signal = self.resample(timestamps, input_signal)

        return resampled_signal

    def apply_polyphase_resampling(self, input_signal):
        """
        Resample the input signal using polyphase resampling.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
        """

        return scipy.signal.resample_poly(input_signal, self.up, self.down)

    def apply_filtering(self, input_signal):
        """
        Create and apply a low pass butterworth filter. The order and the cutoff frequencies of the filter can be specified through the configuration file. Then it applies the butter digital filter forward and backward to the input signal.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter

        .. [2]Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        """

        # Create the low pass butterworth filter
        b, a = scipy.signal.butter(self.order, self.fc / (0.5 * self.acq_frequency))

        # Apply the filter to the input signal
        filtered_signal = scipy.signal.filtfilt(b, a, input_signal)

        return filtered_signal

    def apply_detrending(self, input_signal):
        """
        Detrend the input signal by removing a linear trend or just the mean of the signal.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend.
        """

        return scipy.signal.detrend(input_signal, type=self.detrending_type)

    def resize_data(self, input_signal, balance_board, threshold_1=low_thresh, threshold_2=up_thresh):
        """Remove the beginning and the end of the input signal based on some arbitrary chosen thresholds."""

        return input_signal[threshold_1:threshold_2] if balance_board else\
            input_signal[(threshold_1+self.time_shift):(threshold_2+self.time_shift)]

    def preprocess_signal(self, input_signal, balance_board=False, timestamps=None):
        """
        Pipeline the preprocessing steps.

        The resampling is only applied to the wii balance board data in order to match the force plate acquisition sampling frequency.
        """

        if balance_board:
            if timestamps is not None:
                resampled_signal = self.apply_swarii(input_signal, timestamps)
            else:
                resampled_signal = self.apply_polyphase_resampling(input_signal)

        else:
            resampled_signal = self.apply_polyphase_resampling(input_signal)

        filtered_signal = self.apply_filtering(resampled_signal)
        resized_signal = self.resize_data(filtered_signal, balance_board)
        detrended_data = self.apply_detrending(resized_signal)

        return detrended_data

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

    def preprocess_raw_data(self, data, balance_board=False):
        """Preprocess the raw data"""

        if self.use_swarii and balance_board:
            time_data = data[0]
            relative_timestamps = self.compute_timestamps(time_data)
        else:
            relative_timestamps = None

        if balance_board:
            data = data[1]

        cop_data = self.compute_cop_positions(data, balance_board)
        for key, value in cop_data.items():
            cop_data[key] = self.preprocess_signal(input_signal=value, balance_board=balance_board, timestamps=relative_timestamps)

        return cop_data

    def compute_cop_positions(self, raw_data, balance_board=False):
        """Compute the COP positions in the AP and ML directions."""

        cop_data = {}
        try:
            if balance_board:
                cop_data["COP_x"] = raw_data["Accelerometer"][:, 0]
                cop_data["COP_y"] = raw_data["Accelerometer"][:, 1]
            else:
                cop_data["COP_x"] = self.compute_cop_fp_x(raw_data)
                cop_data["COP_y"] = self.compute_cop_fp_y(raw_data)

            return cop_data

        except Exception:
            raise

    @staticmethod
    def compute_cop_fp_x(data):
        """Compute the y coordinate of the force plate center of pressure (ML direction)."""

        # Force plate height (in mm)
        dz = config["wbb_parameters"]["height"]

        # Force plate sensor values
        Fy1 = data["Fy1"].flatten()
        Mx1 = data["Mx1"].flatten()
        Fz1 = data["Fz1"].flatten()

        Fy1 = pd.DataFrame(Fy1)[0].replace(to_replace=0, value=1).values
        Mx1 = pd.DataFrame(Mx1)[0].replace(to_replace=0, value=1).values
        Fz1 = pd.DataFrame(Fz1)[0].replace(to_replace=0, value=1).values

        cop_fp_x = -(Mx1 - dz * Fy1) / (Fz1)

        return cop_fp_x

    @staticmethod
    def compute_cop_fp_y(data):
        """Compute the x coordinate of the force plate center of pressure (AP direction)."""

        # Force plate height (in mm)
        dz = config["wbb_parameters"]["height"]

        # Force plate sensor values
        Fx1 = data["Fx1"].flatten()
        My1 = data["My1"].flatten()
        Fz1 = data["Fz1"].flatten()

        Fx1 = pd.DataFrame(Fx1)[0].replace(to_replace=0, value=1).values
        My1 = pd.DataFrame(My1)[0].replace(to_replace=0, value=1).values
        Fz1 = pd.DataFrame(Fz1)[0].replace(to_replace=0, value=1).values

        cop_fp_y = -(My1 + dz * Fx1) / (Fz1)

        return cop_fp_y

