# Built-in modules imports
from features import CopFeatures
from utils import load_config

# Third-party module imports
from scipy.signal import welch
from scipy.integrate import cumtrapz
import numpy as np

config = load_config()


class FrequencyFeatures(CopFeatures):
    """Class that implements the frequency domain features derived from the COP positions."""

    fs = config["frequency_features_parameters"]["sampling_frequency"]
    nperseg = config["frequency_features_parameters"]["nperseg"]

    def __init__(self, cop_x, cop_y):
        super(FrequencyFeatures, self).__init__(cop_x, cop_y)
        self.rd_spectral_density = self.compute_rd_power_spectral_density()
        self.ap_spectral_density = self.compute_ap_power_spectral_density()
        self.ml_spectral_density = self.compute_ml_power_spectral_density()
        self.frequency_features = self.compute_frequency_features()

    def compute_power_spectral_density(self, array):
        """
        Function to compute the power spectral density using the scipy implementation of the Welch method.

        References
        ----------
         ..[1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch
        """
        nfft = len(array) * 2
        (f, psd) = welch(array, fs=self.fs, nperseg=self.nperseg, nfft=nfft)

        return (f, psd)

    def compute_rd_power_spectral_density(self):
        """Function to compute the power spectral density of the resultant distance vector of the COP displacement."""

        (f, psd) = self.compute_power_spectral_density(self.cop_rd)

        return (f, psd)

    def compute_ml_power_spectral_density(self):
        """Function to compute the power spectral density of  the COP displacement in the ML direction."""

        (f, psd) = self.compute_power_spectral_density(self.cop_x)

        return (f, psd)

    def compute_ap_power_spectral_density(self):
        """Function to compute the power spectral density of  the COP displacement in the AP direction."""

        (f, psd) = self.compute_power_spectral_density(self.cop_y)

        return (f, psd)

    def compute_rd_power_spectrum_area(self):
        """Function to compute the power spectrum cumulative area of the COP displacement.

        The cumulative integrated area is computed using the composite trapezoidal rule.
        Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumtrapz.html
        """

        (f, psd) = self.rd_spectral_density
        area = cumtrapz(psd, f)

        return area

    def compute_ml_power_spectrum_area(self):
        """Function to compute the power spectrum area of the COP displacement in the ML direction."""

        (f, psd) = self.ml_spectral_density
        area = cumtrapz(psd, f)

        return area

    def compute_ap_power_spectrum_area(self):
        """Function to compute the power spectrum area of the COP displacement in the AP direction."""

        (f, psd) = self.ap_spectral_density
        area = cumtrapz(psd, f)

        return area

    def compute_rd_total_power(self):
        """Function to compute the total power.

        The total power (POWER) is the integrated area of the power spectrum.
        """

        area = self.compute_rd_power_spectrum_area()

        return area[-1]

    def compute_ml_total_power(self):
        """Function to compute the total power in the ML direction."""

        area = self.compute_ml_power_spectrum_area()

        return area[-1]

    def compute_ap_total_power(self):
        """Function to compute the total power in the AP direction."""

        area = self.compute_ap_power_spectrum_area()

        return area[-1]

    def compute_rd_f_peak(self):
        """Function to compute the peak frequency."""

        (f, psd) = self.rd_spectral_density
        p_max_index = psd.argmax()
        f_peak = f[p_max_index]

        return f_peak

    def compute_ml_f_peak(self):
        """Function to compute the peak frequency in the ML direction."""

        (f, psd) = self.ml_spectral_density
        p_max_index = psd.argmax()
        f_peak = f[p_max_index]

        return f_peak

    def compute_ap_f_peak(self):
        """Function to compute the peak frequency in the AP direction."""

        (f, psd) = self.ap_spectral_density
        p_max_index = psd.argmax()
        f_peak = f[p_max_index]

        return f_peak

    def compute_rd_power_frequency(self, n):
        """Function to compute the n% power frequency.

        The n% power frequency is the frequency below which n% of the total power is found.
        """

        power_spectrum_area = self.compute_rd_power_spectrum_area()
        (f, psd) = self.rd_spectral_density
        threshold = (n / 100)
        f_power_index = np.where(power_spectrum_area >= (threshold * power_spectrum_area[-1]))
        f_power = f[f_power_index[0][0]]

        return f_power

    def compute_ml_power_frequency(self, n):
        """Function to compute the n% power frequency in the ML direction."""

        power_spectrum_area = self.compute_ml_power_spectrum_area()
        (f, psd) = self.ml_spectral_density
        threshold = (n / 100)
        f_power_index = np.where(power_spectrum_area >= (threshold * power_spectrum_area[-1]))
        f_power = f[f_power_index[0][0]]

        return f_power

    def compute_ap_power_frequency(self, n):
        """Function to compute the n% power frequency in the AP direction."""

        power_spectrum_area = self.compute_ap_power_spectrum_area()
        (f, psd) = self.ap_spectral_density
        threshold = (n / 100)
        f_power_index = np.where(power_spectrum_area >= (threshold * power_spectrum_area[-1]))
        f_power = f[f_power_index[0][0]]

        return f_power

    def compute_frequency_features(self):
        """Compute all the frequency features and store them in a dictionary."""

        features = {}
        features["Total power-RD"] = self.compute_rd_total_power()
        features["Total power-ML"] = self.compute_ml_total_power()
        features["Total power-AP"] = self.compute_ap_total_power()
        features["Peak frequency-RD"] = self.compute_rd_f_peak()
        features["Peak frequency-ML"] = self.compute_ml_f_peak()
        features["Peak frequency-AP"] = self.compute_ap_f_peak()
        features["50% power frequency-RD"] = self.compute_rd_power_frequency(0.5)
        features["50% power frequency-ML"] = self.compute_ml_power_frequency(0.5)
        features["50% power frequency-AP"] = self.compute_ap_power_frequency(0.5)
        features["80% power frequency-RD"] = self.compute_rd_power_frequency(0.8)
        features["80% power frequency-ML"] = self.compute_ml_power_frequency(0.8)
        features["80% power frequency-AP"] = self.compute_ap_power_frequency(0.8)

        return features

    def summary(self):
        """Print out a summary of the frequency features to standard output."""

        for key, value in self.frequency_features.items():
            print("{}: {}".format(key, value))
