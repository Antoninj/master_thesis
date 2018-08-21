# Third-party module imports
import scipy.signal

# Built-in modules imports
from utils import load_config

config = load_config()


class DataPreprocessor:
    """
    Class that handles the preprocessing related tasks of the acquisition signal using the open source Scipy library.

    More specifically, it relies on the implementations found in the latest release of the scipy signal processing package scipy.signal.

    References
    ----------
    .. [1] Scipy: https://scipy.org
    .. [2] Scipy.signal documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/signal.html
    """

    up = config["preprocessing_parameters"]["upsampling_factor"]
    down = config["preprocessing_parameters"]["downsampling_factor"]
    order = config["preprocessing_parameters"]["filter_order"]
    fc = config["preprocessing_parameters"]["cutoff_frequency"]
    detrending_type = config["preprocessing_parameters"]["detrending_type"]
    threshold = config["preprocessing_parameters"]["cut_threshold"]

    def __init__(self):
        pass

    def apply_resampling(self, input_signal):
        """
        Resample the input signal using polyphase resampling.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
        """

        return scipy.signal.resample_poly(input_signal, self.up, self.down)

    def apply_filtering(self, input_signal, analog_frequency):
        """
        Create and apply a low pass butterworth filter. The order and the cutoff frequencies of the filter can be specified through the configuration file. Then it applies the butter digital filter forward and backward to the input signal.

        References
        ----------
        .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter

        .. [2]Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        """

        # Create the low pass butterworth filter
        b, a = scipy.signal.butter(self.order, self.fc / (0.5 * analog_frequency))

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

    def cut_data(self, input_signal, threshold=None):
        """Cut beginning of the input signal based on a given threshold."""

        if threshold:
            return input_signal[threshold:]
        else:
            return input_signal[self.threshold:]

    def preprocess_sensor_data(self, input_signal, analog_frequency, balance_board=False):
        """
        Pipeline all the preprocessing steps.

        The resampling is only applied to the wii balance board data in order to match the force plate acquisition frequency.
        """

        if balance_board:
            signal = input_signal[:, 0]
            cut_signal = self.cut_data(signal, 500)
            resampled_signal = self.apply_resampling(cut_signal)
            filtered_signal = self.apply_filtering(
                resampled_signal, analog_frequency)
        else:
            signal = input_signal.flatten()
            cut_signal = self.cut_data(signal, 5000)
            filtered_signal = self.apply_filtering(
                cut_signal, analog_frequency)

        return filtered_signal

    def preprocess_raw_data(self, data, frequency, balance_board=False):
        """Preprocess the raw force sensor data"""

        for key, value in data.items():
            data[key] = self.preprocess_sensor_data(value, frequency, balance_board)

        return data

    def preprocess_cop_data(self, data):
        """Preprocess the cop data"""

        for key, value in data.items():
            data[key] = self.apply_detrending(value)

        return data
