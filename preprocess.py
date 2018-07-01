import scipy.signal


class DataPreprocessor(object):
    def __init__(self):
        self.config = self.load_config()

    @staticmethod
    def load_config():
        with open("config/config.json") as cfg:
            config = json.load(cfg)
        return config

    def apply_filtering(self, name, order, data):
        pass

    def apply_resampling(self, data):
        pass

    def apply_detrending(self, data):
        pass
