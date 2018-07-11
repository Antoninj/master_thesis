import numpy as np
from numpy import mean, sqrt, square, diff
from features import CopFeatures
from utils import load_config

config = load_config("process")


class DistanceFeatures(CopFeatures):
    """
    Class that implements the time domain distance features computations derived from the COP positions
    """

    def __init__(self, filepath):
        super(DistanceFeatures, self).__init__(filepath)
        self.distance_features = self.compute_distance_features()

    @staticmethod
    def compute_mean_distance(array):
        """ Helper function to compute the mean value of an array using the numpy mean implementation """

        return array.mean()

    def compute_rd_mean_distance(self):
        """ Compute the average distance from the mean COP """

        return self.compute_mean_distance(self.cop_rd)

    def compute_ml_mean_distance(self):
        """ Compute the average ML distance from the mean COP """

        return self.compute_mean_distance(np.absolute(self.cop_x))

    def compute_ap_mean_distance(self):
        """ Compute the average AP distance from the mean COP """

        return self.compute_mean_distance(np.absolute(self.cop_y))

    @staticmethod
    def compute_rms_distance(array):
        """
        Helper function to compute the root mean square value of an array using the numpy mean, root and square  implementations.
        """

        rms_distance = sqrt(mean(square(array)))
        return rms_distance

    def compute_rd_rms_distance(self):
        """ Compute the root mean square distance from the mean COP """

        return self.compute_rms_distance(self.cop_rd)

    def compute_ml_rms_distance(self):
        """ Compute the root mean square ML distance from the mean COP """

        return self.compute_rms_distance(self.cop_x)

    def compute_ap_rms_distance(self):
        """ Compute the root mean square AP distance from the mean COP """

        return self.compute_rms_distance(self.cop_y)

    @staticmethod
    def compute_path_length(array1, array2):
        distances_1 = diff(array1, axis=0)
        distances_2 = diff(array2, axis=0)
        path_length = np.sqrt((distances_1 ** 2) + (distances_2**2)).sum()

        return path_length

    def compute_rd_path_length(self):
        """ Compute the total length of the COP path """

        return self.compute_path_length(self.cop_x, self.cop_y)

    def compute_ml_path_length(self):
        """ Compute the total length of the COP path in the ML direction """

        distances = np.absolute(diff(self.cop_x, axis=0))
        path_length = distances.sum()

        return path_length

    def compute_ap_path_length(self):
        """ Compute the total length of the COP path in the AP direction """

        distances = np.absolute(np.diff(self.cop_y, axis=0))
        path_length = distances.sum()

        return path_length

    def compute_rd_mean_velocity(self):
        """ Compute the average velocity of the COP """

        mean_velocity = (self.compute_rd_path_length()) / (self.cop_rd.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ml_mean_velocity(self):
        """ Compute the average velocity of the COP in the ML direction """

        mean_velocity = (self.compute_ml_path_length()) / (self.cop_x.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ap_mean_velocity(self):
        """ Compute the average velocity of the COP in the AP direction """

        mean_velocity = (self.compute_ap_path_length()) / (self.cop_y.size / self.acquisition_frequency)
        return mean_velocity

    @staticmethod
    def compute_range(min_value, max_value):
        """
        Helper function to compute the range

        The range is the maximum distance between any two points on the COP path.

        """

        return np.absolute(min_value - max_value)

    def compute_rd_range(self):
        """ Compute the range of the resultant distance time series """

        return self.compute_range(self.cop_rd.min(), self.cop_rd.max())

    def compute_ml_range(self):
        """ Compute the range in the ML direction """

        return self.compute_range(self.cop_x.min(), self.cop_x.max())

    def compute_ap_range(self):
        """ Compute the range in the AP direction """

        return self.compute_range(self.cop_y.min(), self.cop_y.max())

    def compute_distance_features(self):
        """ Function to compute all the distance features and store them in a dictionary """

        features = {}
        features["Rd mean distance"] = self.compute_rd_mean_distance()
        features["ml mean distance"] = self.compute_ml_mean_distance()
        features["ap mean distance"] = self.compute_ap_mean_distance()
        features["Rd rms distance"] = self.compute_rd_rms_distance()
        features["ml rms distance"] = self.compute_ml_rms_distance()
        features["ap rms distance"] = self.compute_ap_rms_distance()
        features["Rd path length"] = self.compute_rd_path_length()
        features["ml path length"] = self.compute_ml_path_length()
        features["ap path length"] = self.compute_ap_path_length()
        features["Rd mean velocity"] = self.compute_rd_mean_velocity()
        features["ml mean velocity"] = self.compute_ml_mean_velocity()
        features["ap mean velocity"] = self.compute_ap_mean_velocity()
        features["Rd range"] = self.compute_rd_range()
        features["ml range"] = self.compute_ml_range()
        features["ap range"] = self.compute_ap_range()

        return features

    def summary(self):
        """ Function to print out a summary of the distance features """

        for key, value in self.distance_features.items():
            print("{}: {}".format(key, value))


class AreaFeatures(DistanceFeatures):
    """
    Class that implements the time domain area features computations derived from the COP positions
    """

    # Constants
    z_05 = config["z_05"]
    F_05 = config["F_05"]

    def __init__(self, filepath):
        super(AreaFeatures, self).__init__(filepath)
        self.area_features = self.compute_area_features()

    def compute_std_rd(self):
        """ Compute the standard deviation of the resultant distance time series """

        std_rd = sqrt(square(self.distance_features["Rd rms distance"]) - square(self.distance_features["Rd mean distance"]))
        return std_rd

    def compute_std_ml(self):
        """ Compute the standard deviation of the ML time series """

        std_ml = sqrt(square(self.distance_features["ml rms distance"]) - square(self.distance_features["ml mean distance"]))
        return std_ml

    def compute_std_ap(self):
        """ Compute the standard deviation of the AP time series  """

        std_ap = sqrt(square(self.distance_features["ap rms distance"]) - square(self.distance_features["ap mean distance"]))
        return std_ap

    def compute_confidence_circle_area(self):
        """
        Function to compute the 95% confidence circle area (AREA-CC)

        It is the area of a circle with a radius equal to the one-sided 95% confidence limit of the RD time series.
        """

        std_rd = self.compute_std_rd()
        area_cc = np.pi * square(self.distance_features["Rd mean distance"] + self.z_05 * std_rd)

        return area_cc

    def compute_confidence_elipse_area(self):
        """
        Function to compute the 95% confidence elipse area (AREA-CE)

        It is the area of the 95% bivariate confidence ellipse, which is expected to enclose approximately 95% of the points on the COP path.
        """

        std_ml = self.compute_std_ml()
        std_ap = self.compute_std_ap()
        area_ce = 2 * np.pi * self.F_05 * sqrt(square(std_ml) * square(std_ap) - np.cov(self.cop_x, self.cop_y)[0][1])

        return area_ce

    def compute_area_features(self):
        """ Function to compute all the area features and store them in a dictionary """

        features = {}
        features["95% confidence circle area"] = self.compute_confidence_circle_area()
        features["95% confidence elipse area"] = self.compute_confidence_elipse_area()

        return features

    def summary(self):
        """ Function to print out a summary of the area features """

        for key, value in self.area_features.items():
            print("{}: {}".format(key, value))


class HybridFeatures(AreaFeatures):
    """
    Class that implements the time domain hybrid features computations derived from the COP positions
    """

    def __init__(self, filepath):
        super(HybridFeatures, self).__init__(filepath)
        self.hybrid_features = self.compute_hybrid_features()

    def compute_sway_area(self):
        """
        Function that computes the sway area

        Sway area estimates the area enclosed by the COP path per unit of time.
        """

        sway_values = []
        T = (self.cop_rd.size / self.acquisition_frequency)
        for i in range(len(self.cop_rd) - 1):
            sway_values.append((self.cop_x[i + 1] * self.cop_y[i] - self.cop_x[i] * self.cop_y[i + 1]) / (2 * T))

        sway_area = np.array(sway_values).sum()

        return sway_area

    def compute_mean_frequency(self):
        """
        Function to compute the mean frequency

        The mean frequency (MFREQ) is the rotational frequency,in revolutions per second or Hz, of the COP if it had traveled the total excursions around a circle with a radius of the mean distance.
        """

        mean_frequency = (self.distance_features["Rd mean velocity"]) / (2 * np.pi * self.distance_features["Rd mean distance"])

        return mean_frequency

    def compute_mean_frequency_ml(self):
        """
        Function to compute the mean frequency in the ML direction

        The mean frequency-ML is the frequency, in Hz, of a sinusoidal oscillation with an average value of the mean distance-ML and a total path length of total excursions-ML.
        """

        mean_frequency_ml = (self.distance_features["ml mean velocity"]) / (4 * sqrt(2) * self.distance_features["ml mean distance"])

        return mean_frequency_ml

    def compute_mean_frequency_ap(self):
        """
        Function to compute the mean frequency in the AP direction

        The mean frequency-AP is the frequency, in Hz, of a sinusoidal oscillation with an average value of the mean distance-AP and a total path length of total excursions-AP.
        """

        mean_frequency_ap = (self.distance_features["ap mean velocity"]) / (4 * sqrt(2) * self.distance_features["ap mean distance"])

        return mean_frequency_ap

    def compute_fractal_dimension(self, d):
        """
        Function to compute the fractal dimension (FD)

        The fractal dimension (FD) is a unitless measure of the degree to which a curve fills the metric space which it encompasses.
        """

        N = self.cop_rd.size
        FD = (np.log(N)) / (np.log((N * d) / (self.distance_features["Rd path length"])))

        return FD

    def compute_fractal_dimension_cc(self):
        """
        Function to compute the fractal dimension-CC

        Fractal dimension-CC is based on the 95% confidence circle.
        """

        std_rd = self.compute_std_rd()
        d_fd_cc = 2 * (self.distance_features["Rd mean distance"] + self.z_05 * std_rd)

        return self.compute_fractal_dimension(d_fd_cc)

    def compute_fractal_dimension_ce(self):
        """
        Function to compute the fractal dimension-CE

        Fractal dimension-CE is based on the 95% confidence elipse.
        """

        std_ml = self.compute_std_ml()
        std_ap = self.compute_std_ap()
        d_fd_ce = sqrt(8 * self.F_05 * sqrt(square(std_ml) * square(std_ap) - np.cov(self.cop_x, self.cop_y)[0][1]))

        return self.compute_fractal_dimension(d_fd_ce)

    def compute_hybrid_features(self):
        """ Function to compute all the hybrid features and store them in a dictionary """

        features = {}
        features["Sway area"] = self.compute_sway_area()
        features["Mean frequency"] = self.compute_mean_frequency()
        features["Mean frequency-ml"] = self.compute_mean_frequency_ml()
        features["Mean frequency-ap"] = self.compute_mean_frequency_ap()
        features["Fractal dimension-CC"] = self.compute_fractal_dimension_cc()
        features["Fractal dimension-CE"] = self.compute_fractal_dimension_ce()
        return features

    def summary(self):
        """ Function to print out a summary of the hybrid features """

        for key, value in self.hybrid_features.items():
            print("{}: {}".format(key, value))
