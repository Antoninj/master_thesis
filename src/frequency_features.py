# Built-in modules imports
from features import CopFeatures
from utils import load_config
from mtspec import mtspec

# Third-party module imports
from scipy.signal import welch
from scipy.integrate import cumtrapz
import numpy as np
import spectrum


config = load_config()


class FrequencyFeatures(CopFeatures):
    """Class that implements the frequency domain features derived from the COP positions."""

    psd_method_used = config["frequency_features_parameters"]["psd_method"]
    fs = config["frequency_features_parameters"]["sampling_frequency"]
    frequency_range = config["frequency_features_parameters"]["frequency_range"]

    # Welch periodogram parameters
    nperseg = config["frequency_features_parameters"]["welch"]["nperseg"]

    # Adaptive weighted multitaper spectrum parameters
    delta = config["frequency_features_parameters"]["multitaper"]["delta"]
    time_bandwidth = config["frequency_features_parameters"]["multitaper"]["time_bandwidth"]
    number_of_tapers = config["frequency_features_parameters"]["multitaper"]["number_of_tapers"]



    def __init__(self, cop_x, cop_y):
        super(FrequencyFeatures, self).__init__(cop_x, cop_y)

        self.psd_methods_dict = self.create_psd_methods_dict()
        default_psd = "welch"
        self.psd_method_impl = self.psd_methods_dict.get(self.psd_method_used, self.psd_methods_dict[default_psd])

        self.rd_spectral_density = self.psd_method_impl(self.cop_rd)
        self.ap_spectral_density = self.psd_method_impl(self.cop_x)
        self.ml_spectral_density = self.psd_method_impl(self.cop_y)
        self.frequency_features = self.compute_frequency_features()

    def create_psd_methods_dict(self):
        psd_methods = {}
        psd_methods["welch"] = self.compute_welch_psd
        psd_methods["multitaper"] = self.compute_multitaper_psd
        psd_methods["multitaper2"] = self.compute_multitaper_bis_psd
        psd_methods["ma"] = self.compute_ma_psd
        psd_methods["arma"] = self.compute_arma_psd
        psd_methods["yule"] = self.compute_yulewalker_psd
        psd_methods["burg"] = self.compute_burg_psd
        psd_methods["covar"] = self.compute_covar_psd
        psd_methods["modcovar"] = self.compute_modcovar_psd

        return psd_methods

    def compute_multitaper_psd(self, array):
        """
        Estimate the adaptive weighted multitaper spectrum, as in Thomson 1982. This is done by estimating the DPSS
        (discrete prolate spheroidal sequences), multiplying each of the tapers with the data series, take the FFT,
        and using the adaptive scheme for a better estimation. It outputs the power spectral density (PSD).

        References
        ----------
         ..[1] mtspec package documentation: http://krischer.github.io/mtspec/multitaper_mtspec.html
        """

        psd, f = mtspec(data=array, delta=self.delta, time_bandwidth=self.time_bandwidth, number_of_tapers=self.number_of_tapers)

        psd = psd[self.frequency_range[0]: self.frequency_range[1]]
        f = f[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_welch_psd(self, array):
        """
        Function to compute the power spectral density using the scipy implementation of the Welch method.

        References
        ----------
         ..[1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch
        """

        nfft = len(array)
        (f, psd) = welch(array, fs=self.fs, nperseg=self.nperseg, nfft=nfft)

        psd = psd[self.frequency_range[0]: self.frequency_range[1]]
        f = f[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_multitaper_bis_psd(self, array):
        p = spectrum.mtm.pmtm(array, NW=self.time_bandwidth, k=self.number_of_tapers, NFFT=len(array))
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_ma_psd(self, array):
        p = spectrum.pma(array, 15, 30, NFFT=len(array), sampling=self.fs)
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_arma_psd(self, array):
        p = spectrum.parma(array, 15, 15, 30, NFFT=len(array), sampling=self.fs)
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_yulewalker_psd(self, array):
        p = spectrum.pyule(array, 15, norm="biased", NFFT=len(array))
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_burg_psd(self, array):
        p = spectrum.pburg(array, order=15, NFFT=len(array))
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_covar_psd(self, array):
        p = spectrum.pcovar(array, 15, NFFT=len(array))
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    def compute_modcovar_psd(self, array):
        p = spectrum.pmodcovar(array, 15, NFFT=len(array))
        p.run()
        psd = p.psd[self.frequency_range[0]: self.frequency_range[1]]
        f = p.frequencies()[self.frequency_range[0]: self.frequency_range[1]]

        return (f, psd)

    @staticmethod
    def compute_power_spectral_density_area(power_spectrum):
        """Function to compute the power spectral density cumulative area.

        The cumulative integrated area is computed using the composite trapezoidal rule.
        Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumtrapz.html
        """

        (f, psd) = power_spectrum
        return cumtrapz(psd, f)

    def compute_rd_power_spectral_density_area(self):
        """Function to compute the resultant distance displacement power spectral density cumulative area."""

        return self.compute_power_spectral_density_area(self.rd_spectral_density)

    def compute_ml_power_spectral_density_area(self):
        """Function to compute the power spectral density area of the COP displacement in the ML direction."""

        return self.compute_power_spectral_density_area(self.ml_spectral_density)

    def compute_ap_power_spectral_density_area(self):
        """Function to compute the power spectral density area of the COP displacement in the AP direction."""

        return self.compute_power_spectral_density_area(self.ap_spectral_density)


    def compute_rd_total_power(self):
        """Function to compute the power spectral density total power.

        The total power (POWER) is the integrated area of the power spectrum.
        """

        area = self.compute_rd_power_spectral_density_area()

        return area[-1]

    def compute_ml_total_power(self):
        """Function to compute the total power in the ML direction."""

        area = self.compute_ml_power_spectral_density_area()

        return area[-1]

    def compute_ap_total_power(self):
        """Function to compute the total power in the AP direction."""

        area = self.compute_ap_power_spectral_density_area()

        return area[-1]

    @staticmethod
    def compute_peak_frequency(power_spectrum):
        """Function to compute the peak frequency."""

        (f, psd) = power_spectrum
        p_max_index = psd.argmax()
        f_peak = f[p_max_index]

        return f_peak

    def compute_rd_f_peak(self):
        """Function to compute the resultant distance power spectral density peak frequency."""

        return self.compute_peak_frequency(self.rd_spectral_density)

    def compute_ml_f_peak(self):
        """Function to compute the peak frequency in the ML direction."""

        return self.compute_peak_frequency(self.ml_spectral_density)

    def compute_ap_f_peak(self):
        """Function to compute the peak frequency in the AP direction."""

        return self.compute_peak_frequency(self.ap_spectral_density)

    @staticmethod
    def compute_power_frequency(n, power_spectrum_area, power_spectrum):
        """Function to compute the n% power frequency.

        The n% power frequency is the frequency below which n% of the total power is found.
        """

        (f, psd) = power_spectrum
        threshold = (n / 100)
        f_power_index = np.where(power_spectrum_area >= (threshold * power_spectrum_area[-1]))
        f_power = f[f_power_index[0][0]]

        return f_power

    def compute_rd_power_frequency(self, n):
        """Function to compute the resultant distance power spectral density n% power frequency."""

        power_spectrum_area = self.compute_rd_power_spectral_density_area()
        (f, psd) = self.rd_spectral_density

        return self.compute_power_frequency(n, power_spectrum_area, (f, psd))

    def compute_ml_power_frequency(self, n):
        """Function to compute the n% power frequency in the ML direction."""

        power_spectrum_area = self.compute_ml_power_spectral_density_area()
        (f, psd) = self.ml_spectral_density

        return self.compute_power_frequency(n, power_spectrum_area, (f, psd))

    def compute_ap_power_frequency(self, n):
        """Function to compute the n% power frequency in the AP direction."""

        power_spectrum_area = self.compute_ap_power_spectral_density_area()
        (f, psd) = self.ap_spectral_density

        return self.compute_power_frequency(n, power_spectrum_area, (f, psd))

    def compute_spectral_moment(self, k, psd_estimate):

        delta_f = self.delta
        spectral_moment = sum(
            [np.power(((m + self.frequency_range[0]) * delta_f), k) * psd for m, psd in enumerate(psd_estimate)])

        return spectral_moment

    @staticmethod
    def compute_centroidal_frequency(mu_0, mu_2):
        """" Function to compute the centroidal frequency.

        The centroidal frequency (CFREQ) is the frequency at which the spectral mass is concentrated,
        which is the square root of the ratio of the second to the zeroth spectral moments."""

        return np.sqrt(mu_2 / mu_0)

    def compute_rd_centroidal_frequency(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.rd_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.rd_spectral_density[1])

        return self.compute_centroidal_frequency(mu_0, mu_2)

    def compute_ml_centroidal_frequency(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.ml_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.ml_spectral_density[1])

        return self.compute_centroidal_frequency(mu_0, mu_2)

    def compute_ap_centroidal_frequency(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.ap_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.ap_spectral_density[1])

        return self.compute_centroidal_frequency(mu_0, mu_2)

    @staticmethod
    def compute_frequency_dispersion(mu_0, mu_1, mu_2):
        """ Function to compute the frequency dispersion.

        The frequency dispersion (FREQD) is a unitless measure of the variability in the frequency content of the power spectral density.
        """
        return np.sqrt(1 - (np.square(mu_1)) / (mu_0 * mu_2))

    def compute_rd_frequency_dispersion(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.rd_spectral_density[1])
        mu_1 = self.compute_spectral_moment(1, self.rd_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.rd_spectral_density[1])

        return self.compute_frequency_dispersion(mu_0, mu_1, mu_2)

    def compute_ml_frequency_dispersion(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.ml_spectral_density[1])
        mu_1 = self.compute_spectral_moment(1, self.ml_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.ml_spectral_density[1])

        return self.compute_frequency_dispersion(mu_0, mu_1, mu_2)

    def compute_ap_frequency_dispersion(self):
        """Function to compute the resultant distance power spectral density centroidal frequency."""

        mu_0 = self.compute_spectral_moment(0, self.ap_spectral_density[1])
        mu_1 = self.compute_spectral_moment(1, self.ap_spectral_density[1])
        mu_2 = self.compute_spectral_moment(2, self.ap_spectral_density[1])

        return self.compute_frequency_dispersion(mu_0, mu_1, mu_2)

    def compute_frequency_features(self):
        """Compute all the frequency features and store them in a dictionary."""

        features = {}
        features["Total power-RD"] = self.compute_rd_total_power()
        features["Total power-ML"] = self.compute_ml_total_power()
        features["Total power-AP"] = self.compute_ap_total_power()
        features["Peak frequency-RD"] = self.compute_rd_f_peak()
        features["Peak frequency-ML"] = self.compute_ml_f_peak()
        features["Peak frequency-AP"] = self.compute_ap_f_peak()
        features["50% power frequency-RD"] = self.compute_rd_power_frequency(50)
        features["50% power frequency-ML"] = self.compute_ml_power_frequency(50)
        features["50% power frequency-AP"] = self.compute_ap_power_frequency(50)
        features["80% power frequency-RD"] = self.compute_rd_power_frequency(80)
        features["80% power frequency-ML"] = self.compute_ml_power_frequency(80)
        features["80% power frequency-AP"] = self.compute_ap_power_frequency(80)
        features["Centroidal frequency-RD"] = self.compute_rd_centroidal_frequency()
        features["Centroidal frequency-ML"] = self.compute_ml_centroidal_frequency()
        features["Centroidal frequency-AP"] = self.compute_ap_centroidal_frequency()
        features["Frequency dispersion-RD"] = self.compute_rd_frequency_dispersion()
        features["Frequency dispersion-ML"] = self.compute_ml_frequency_dispersion()
        features["Frequency dispersion-AP"] = self.compute_ap_frequency_dispersion()

        return features

    def summary(self):
        """Print out a summary of the frequency features to standard output."""

        for key, value in self.frequency_features.items():
            print("{}: {}".format(key, value))
