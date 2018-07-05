import scipy.signal
import json
from utils import load_config


class DataPreprocessor(object):
    """ Class that handles the preprocessing related tasks of the acquisition signal using the open source Scipy library.

    More specifically, it relies on the implementations found in the latest release of the scipy signal processing package scipy.signal. """

    def __init__(self):
        self.config = self.load_config()

    @staticmethod
    def load_config():
        with open("config/preprocess.json") as cfg:
            config = json.load(cfg)
        return config

    def apply_resampling(self, input_signal):
        """Resample the input signal using polyphase filtering.

        Scipy documentation: https: // docs.scipy.org / doc / scipy - 1.1.0 / reference / generated / scipy.signal.resample_poly.html  # scipy.signal.resample_poly """
        up = self.config["upsampling factor"]
        down = self.config["downsampling factor"]
        return scipy.signal.resample_poly(input_signal, up, down)

    def apply_filtering(self, input_signal, analog_frequency):
        """ This function creates a low pass butterworth filter. The order and the cutoff frequencies of the filter can be specified through the configuration file.

        Scipy documentation: https: // docs.scipy.org / doc / scipy - 1.1.0 / reference / generated / scipy.signal.butter.html  # scipy.signal.butter

        Then it applies the butter digital filter forward and backward to the input signal

        Scipy documentation: https: // docs.scipy.org / doc / scipy / reference / generated / scipy.signal.filtfilt.html """

        # Retrieve the order and cutoff frequency parameters for the filter from the configuration file
        order = self.config["order"]
        fc = self.config["cutoff frequency"]

        # Create the filter
        b, a = scipy.signal.butter(order, fc / (0.5 * analog_frequency))

        # Apply the filter
        filtered_signal = scipy.signal.filtfilt(b, a, input_signal)

        return filtered_signal

    def apply_detrending(self, input_signal):
        """Detrend the input signal by removing a linear trend or just the mean of the signal.

        Scipy documentation: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend """

        detrending_type = self.config["detrending type"]

        return scipy.signal.detrend(input_signal, type=detrending_type)

    def preprocess(self, input_signal, analog_frequency, balance_board=False):
        """ Wrapper function that applies all the preprocessing steps at once. """

        if balance_board:
            resampled_signal = self.apply_resampling(input_signal)
            filtered_signal = self.apply_filtering(
                resampled_signal, analog_frequency)
        else:
            filtered_signal = self.apply_filtering(
                input_signal, analog_frequency)
        preprocessed_signal = self.apply_detrending(filtered_signal)

        return preprocessed_signal
