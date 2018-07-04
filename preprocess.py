import scipy.signal
import json


class DataPreprocessor(object):
    def __init__(self):
        self.config = self.load_config()

    @staticmethod
    def load_config():
        with open("config/preprocess.json") as cfg:
            config = json.load(cfg)
        return config

    def apply_filtering(self, data, analog_frequency):

        order = self.config["order"]
        fc = self.config["cutoff_frequency"]
        b, a = scipy.signal.butter(order, fc / (0.5 * analog_frequency))
        filter_data = scipy.signal.filtfilt(b, a, data[:, 0])

        return filter_data

    def apply_resampling(self, data):
        pass

    def apply_detrending(self, data):
        return scipy.signal.detrend(data, type="constant")
