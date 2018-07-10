import numpy as np
from features import CopFeatures
from scipy.signal import welch


class FrequencyFeatures(CopFeatures):
    """ Class that implements the frequency domain features derived from the COP positions """

    def __init__(self, filepath):
        super(FrequencyFeatures, self).__init__(filepath)
        self.spectral_density = self.compute_rd_power_spectral_density()
        #self.frequency_features = self.compute_frequency_features()

    def compute_rd_power_spectral_density(self):
        fs = 10
        x = self.cop_rd
        nfft = round(x.size / 2)
        f, psd = welch(x, fs=fs, nfft=nfft)

        return (f, psd)
