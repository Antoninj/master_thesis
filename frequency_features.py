import numpy as np
from features import CopFeatures
from scipy.signal import welch


class FrequencyFeatures(CopFeatures):
    """                             """

    def __init__(self, filepath):
        super(FrequencyFeatures, self).__init__(filepath)
        self.spectral_density = self.compute_power_sprectral_density()
        self.frequency_features = self.compute_frequency_features()

