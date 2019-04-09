# Third-party module imports
import logging
import warnings

import numpy as np
import pandas as pd
import scipy.signal
from resampling import SWARII
# Built-in modules imports
from utils import load_config

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

    fp_up = config["preprocessing_parameters"]["fp_upsampling_factor"]
    fp_down = config["preprocessing_parameters"]["fp_downsampling_factor"]
    order = config["preprocessing_parameters"]["filter_order"]
    fc = config["preprocessing_parameters"]["cutoff_frequency"]
    detrending_type = config["preprocessing_parameters"]["detrending_type"]
    low_thresh = config["preprocessing_parameters"]["time_window_lower_threshold"]
    up_thresh = config["preprocessing_parameters"]["time_window_upper_threshold"]
    time_shift = config["preprocessing_parameters"]["time_shift"]
    swarii_window = config["preprocessing_parameters"]["swarii_window_size"]
    acq_frequency = config["preprocessing_parameters"]["acquisition_frequency"]
    use_swarii = config["preprocessing_parameters"]["apply_swarii"]

    def __init__(self):
        super(DataPreprocessor, self).__init__(window_size=self.swarii_window, desired_frequency=self.acq_frequency)

    def apply_swarii_resampling(self, input_signal, timestamps):
        """Apply the SWARII to resample a given signal."""

        resampled_time, resampled_signal = self.resample(timestamps, input_signal)

        return resampled_signal

    def apply_resampling(self, input_signal, num):
        """
        Resample the input signal using Fourier method.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html#scipy.signal.resample
         """

        return scipy.signal.resample(input_signal, num)

    def apply_downsampling(self, input_signal, down):
        """
        Downsample the input signal.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
        """

        return scipy.signal.decimate(input_signal, down)

    def apply_polyphase_resampling(self, input_signal, up, down):
        """
        Resample the input signal using polyphase resampling.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
        """

        return scipy.signal.resample_poly(input_signal, up, down)

    def apply_filtering(self, input_signal):
        """
        Create and apply a low pass butterworth filter. The order and the cutoff frequencies of the filter can be specified through the configuration file.
        The butter digital filter is applied forward and backward to the input signal.

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

    def apply_reframing(self, input_signal, balance_board, threshold_1=low_thresh, threshold_2=up_thresh):
        """Remove the beginning and the end of the input signal based on some arbitrary chosen thresholds and remove the acquisition time shift between the two devices."""

        return input_signal[threshold_1:threshold_2] if balance_board else\
            input_signal[(threshold_1+self.time_shift):(threshold_2+self.time_shift)]

    def preprocess_signal(self, input_signal, balance_board=False, timestamps=None):
        """
        Pipeline the preprocessing steps.

        The SWARII resampling is applied to the wii balance board data in order to match the downsampled force plate sampling frequency (using polyphase resampling).

        NB: The WBB signal doesn't need to be filtered "as doing averages over a window already denoises the signal" (cf SWARII paper).
        """

        if balance_board:
            if self.use_swarii:
                # Resample the balance board data using SWARII
                resampled_signal = self.apply_swarii_resampling(input_signal, timestamps)
                reframed_data = self.apply_reframing(resampled_signal, balance_board)

            else:
                # Resample the balance board data using fourier resampling
                acquisition_duration = timestamps[-1]
                num = round(acquisition_duration * self.acq_frequency)
                resampled_signal = self.apply_resampling(input_signal, num)
                filtered_signal = self.apply_filtering(resampled_signal)
                reframed_data = self.apply_reframing(filtered_signal, balance_board)

        else:
            # Downsample the force plate data using decimation
            resampled_signal = self.apply_downsampling(input_signal, self.fp_down)
            filtered_signal = self.apply_filtering(resampled_signal)
            reframed_data = self.apply_reframing(filtered_signal, balance_board)

        detrended_data = self.apply_detrending(reframed_data)

        return detrended_data

    def preprocess_raw_data(self, data, balance_board=False):
        """Preprocess the raw data"""

        if balance_board:
            relative_timestamps = data[0]
            data = data[1]
        else:
            relative_timestamps = None

        cop_data = self.compute_cop_positions(data, balance_board)
        for key, value in cop_data.items():
            cop_data[key] = self.preprocess_signal(input_signal=value,
                                                   balance_board=balance_board, timestamps=relative_timestamps)

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

    def compute_cop_fp_x(self, data):
        """Compute the y coordinate of the force plate center of pressure (ML direction)."""

        # Force plate height (in mm)
        dz = config["force_plate_parameters"]["dz"]

        # Force plate sensor values
        Fy1 = data["Fy1"].flatten()
        Mx1 = data["Mx1"].flatten()
        Fz1 = data["Fz1"].flatten()

        # Replace null data with previous/following non null values
        Fy1 = self.replace_missing_data(Fy1)
        Mx1 = self.replace_missing_data(Mx1)
        Fz1 = self.replace_missing_data(Fz1)

        cop_fp_x = -(Mx1 - dz * Fy1) / (Fz1)

        return cop_fp_x

    def compute_cop_fp_y(self, data):
        """Compute the x coordinate of the force plate center of pressure (AP direction)."""

        # Force plate height (in mm)
        dz = config["force_plate_parameters"]["dz"]

        # Force plate sensor values
        Fx1 = data["Fx1"].flatten()
        My1 = data["My1"].flatten()
        Fz1 = data["Fz1"].flatten()

        # Replace null data with previous or next non null values
        Fx1 = self.replace_missing_data(Fx1)
        My1 = self.replace_missing_data(My1)
        Fz1 = self.replace_missing_data(Fz1)

        cop_fp_y = -(My1 + (dz * Fx1)) / (Fz1)

        return cop_fp_y

    @staticmethod
    def replace_missing_data(data):
        # TODO : refine this process!
        forward_fill_missing_data = pd.DataFrame(data)[0].replace(to_replace=0, method='ffill').values
        backward_fill_missing_data = pd.DataFrame(forward_fill_missing_data)[0].replace(to_replace=0,
                                                                                        method='bfill').values
        return backward_fill_missing_data
