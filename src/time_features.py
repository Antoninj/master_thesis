# Built-in modules imports
from features import CopFeatures
from utils import load_config

# Third-party modules imports
import numpy as np
import warnings

config = load_config()

# Set numpy error level to warning
# np.seterr(all='warn')
# warnings.filterwarnings('error')


class DistanceFeatures(CopFeatures):
    """
    Class that implements the time domain distance features computations derived from the COP positions.
    """

    def __init__(self, cop_x, cop_y):
        super(DistanceFeatures, self).__init__(cop_x, cop_y)
        self.distance_features = self.compute_distance_features()

    @staticmethod
    def compute_mean_distance(array):
        """Compute the mean value of an array using the numpy mean implementation."""

        return array.mean()

    def compute_rd_mean_distance(self):
        """Compute the average distance from the mean COP."""

        return self.compute_mean_distance(self.cop_rd)

    def compute_ml_mean_distance(self):
        """Compute the average ML distance from the mean COP."""

        return self.compute_mean_distance(np.absolute(self.cop_x))

    def compute_ap_mean_distance(self):
        """Compute the average AP distance from the mean COP."""

        return self.compute_mean_distance(np.absolute(self.cop_y))

    @staticmethod
    def compute_rms_distance(array):
        """
        Compute the root mean square value of an array using the numpy mean, root and square implementations.
        """

        rms_distance = np.sqrt(np.mean(np.square(array)))
        return rms_distance

    def compute_rd_rms_distance(self):
        """Compute the root mean square distance from the mean COP."""

        return self.compute_rms_distance(self.cop_rd)

    def compute_ml_rms_distance(self):
        """Compute the root mean square ML distance from the mean COP."""

        return self.compute_rms_distance(self.cop_x)

    def compute_ap_rms_distance(self):
        """Compute the root mean square AP distance from the mean COP."""

        return self.compute_rms_distance(self.cop_y)

    @staticmethod
    def compute_path_length(array1, array2):
        distances_1 = np.diff(array1, axis=0)
        distances_2 = np.diff(array2, axis=0)
        path_length = np.sqrt((distances_1 ** 2) + (distances_2**2)).sum()

        return path_length

    def compute_rd_path_length(self):
        """Compute the total length of the COP path."""

        return self.compute_path_length(self.cop_x, self.cop_y)

    def compute_ml_path_length(self):
        """Compute the total length of the COP path in the ML direction."""

        distances = np.absolute(np.diff(self.cop_x, axis=0))
        path_length = distances.sum()

        return path_length

    def compute_ap_path_length(self):
        """Compute the total length of the COP path in the AP direction."""

        distances = np.absolute(np.diff(self.cop_y, axis=0))
        path_length = distances.sum()

        return path_length

    def compute_rd_mean_velocity(self):
        """Compute the average velocity of the COP."""

        mean_velocity = (self.compute_rd_path_length()) / \
            (self.cop_rd.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ml_mean_velocity(self):
        """Compute the average velocity of the COP in the ML direction."""

        mean_velocity = (self.compute_ml_path_length()) / \
            (self.cop_x.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ap_mean_velocity(self):
        """Compute the average velocity of the COP in the AP direction."""

        mean_velocity = (self.compute_ap_path_length()) / \
            (self.cop_y.size / self.acquisition_frequency)
        return mean_velocity

    @staticmethod
    def compute_range(min_value, max_value):
        """
        Compute the range.

        The range is the maximum distance between any two points on the COP path.
        """

        return np.absolute(min_value - max_value)

    def compute_rd_range(self):
        """Compute the range of the resultant distance time series."""

        return self.compute_range(self.cop_rd.min(), self.cop_rd.max())

    def compute_ml_range(self):
        """Compute the range in the ML direction."""

        return self.compute_range(self.cop_x.min(), self.cop_x.max())

    def compute_ap_range(self):
        """Compute the range in the AP direction."""

        return self.compute_range(self.cop_y.min(), self.cop_y.max())

    def compute_distance_features(self):
        """Compute all the distance features and store them in a dictionary."""

        features = {}
        features["Mean distance"] = self.compute_rd_mean_distance()
        features["Mean distance-ML"] = self.compute_ml_mean_distance()
        features["Mean distance-AP"] = self.compute_ap_mean_distance()
        features["Rms distance"] = self.compute_rd_rms_distance()
        features["Rms distance-ML"] = self.compute_ml_rms_distance()
        features["Rms distance-AP"] = self.compute_ap_rms_distance()
        features["Path length"] = self.compute_rd_path_length()
        features["Path length-ML"] = self.compute_ml_path_length()
        features["Path length-AP"] = self.compute_ap_path_length()
        features["Mean velocity"] = self.compute_rd_mean_velocity()
        features["Mean velocity-ML"] = self.compute_ml_mean_velocity()
        features["Mean velocity-AP"] = self.compute_ap_mean_velocity()
        features["Range"] = self.compute_rd_range()
        features["Range-ML"] = self.compute_ml_range()
        features["Range-AP"] = self.compute_ap_range()

        return features

    def summary(self):
        """Print out a summary of the distance features to standard output."""

        for key, value in self.distance_features.items():
            print("{}: {}".format(key, value))


class AreaFeatures(DistanceFeatures):
    """
    Class that implements the time domain area features computations derived from the COP positions.
    """

    # Constants
    z_05 = config["time_features_parameters"]["z_05"]
    F_05 = config["time_features_parameters"]["F_05"]

    def __init__(self, cop_x, cop_y):
        super(AreaFeatures, self).__init__(cop_x, cop_y)
        self.area_features = self.compute_area_features()

    def compute_std_rd(self):
        """Compute the standard deviation of the resultant distance time series."""

        std_rd = np.sqrt(np.square(self.distance_features["Rms distance"]) - np.square(
            self.distance_features["Rms distance"]))
        return std_rd

    def compute_std_ml(self):
        """Compute the standard deviation of the ML time series."""

        std_ml = np.sqrt(np.square(self.distance_features["Rms distance-ML"]) - np.square(
            self.distance_features["Rms distance-ML"]))
        return std_ml

    def compute_std_ap(self):
        """Compute the standard deviation of the AP time series."""

        std_ap = np.sqrt(np.square(self.distance_features["Rms distance-AP"]) - np.square(
            self.distance_features["Rms distance-AP"]))
        return std_ap

    def compute_confidence_circle_area(self):
        """
        Compute the 95% confidence circle area (AREA-CC).

        It is the area of a circle with a radius equal to the one-sided 95% confidence limit of the RD time series.
        """

        std_rd = self.compute_std_rd()
        area_cc = np.pi * \
            np.square(
                self.distance_features["Mean distance"] + self.z_05 * std_rd)

        return area_cc

    def compute_confidence_elipse_area(self):
        """
        Compute the 95% confidence elipse area (AREA-CE).

        It is the area of the 95% bivariate confidence ellipse, which is expected to enclose approximately 95% of the points on the COP path.
        """

        std_ml = self.compute_std_ml()
        std_ap = self.compute_std_ap()

        # For debuging purposes
        # print(square(square(std_ml)) + square(square(std_ap)) + 6 * square(std_ml) * square(std_ap) -
        #     4 * square(np.cov(self.cop_x, self.cop_y)[0][1]) - (square(std_ml) + square(std_ap)))

        try:
            area_ce = np.pi * self.F_05 * np.sqrt(np.square(np.square(std_ml)) + np.square(np.square(std_ap)) + 6 * np.square(
                std_ml) * np.square(std_ap) - 4 * np.square(np.cov(self.cop_x, self.cop_y)[0][1]) - (np.square(std_ml) + np.square(std_ap)))

            return area_ce

        except Warning:
            return np.nan

    def compute_area_features(self):
        """Compute all the area features and store them in a dictionary."""

        features = {}
        features["95% confidence circle area"] = self.compute_confidence_circle_area()
        features["95% confidence elipse area"] = self.compute_confidence_elipse_area()

        return features

    def summary(self):
        """Print out a summary of the area features to standard output."""

        for key, value in self.area_features.items():
            print("{}: {}".format(key, value))


class HybridFeatures(AreaFeatures):
    """
    Class that implements the time domain hybrid features computations derived from the COP positions.
    """

    def __init__(self, cop_x, cop_y):
        super(HybridFeatures, self).__init__(cop_x, cop_y)
        self.hybrid_features = self.compute_hybrid_features()

    def compute_sway_area(self):
        """
        Compute the sway area.

        Sway area estimates the area enclosed by the COP path per unit of time.
        """

        sway_values = []
        T = (self.cop_rd.size / self.acquisition_frequency)
        for i in range(len(self.cop_rd) - 1):
            sway_values.append(np.absolute(
                self.cop_x[i + 1] * self.cop_y[i] - self.cop_x[i] * self.cop_y[i + 1]) / (2 * T))

        sway_area = np.array(sway_values).sum()

        return sway_area

    def compute_mean_frequency(self):
        """
        Compute the mean frequency.

        The mean frequency (MFREQ) is the rotational frequency,in revolutions per second or Hz, of the COP if it had traveled the total excursions around a circle with a radius of the mean distance.
        """

        mean_frequency = (self.distance_features["Mean velocity"]) / (
            2 * np.pi * self.distance_features["Mean distance"])

        return mean_frequency

    def compute_mean_frequency_ml(self):
        """
        Compute the mean frequency in the ML direction.

        The mean frequency-ML is the frequency, in Hz, of a sinusoidal oscillation with an average value of the mean distance-ML and a total path length of total excursions-ML.
        """

        mean_frequency_ml = (self.distance_features["Mean velocity-ML"]) / (
            4 * np.sqrt(2) * self.distance_features["Mean distance-ML"])

        return mean_frequency_ml

    def compute_mean_frequency_ap(self):
        """
        Compute the mean frequency in the AP direction.

        The mean frequency-AP is the frequency, in Hz, of a sinusoidal oscillation with an average value of the mean distance-AP and a total path length of total excursions-AP.
        """

        mean_frequency_ap = (self.distance_features["Mean velocity-AP"]) / (
            4 * np.sqrt(2) * self.distance_features["Mean distance-AP"])

        return mean_frequency_ap

    def compute_fractal_dimension(self, d):
        """
        Compute the fractal dimension (FD).

        The fractal dimension (FD) is a unitless measure of the degree to which a curve fills the metric space which it encompasses.
        """

        N = self.cop_rd.size
        FD = (np.log(N)) / \
            (np.log((N * d) / (self.distance_features["Path length"])))

        return FD

    def compute_fractal_dimension_cc(self):
        """
        Compute the fractal dimension-CC.

        The fractal dimension-CC is based on the 95% confidence circle.
        """

        std_rd = self.compute_std_rd()
        d_fd_cc = 2 * \
            (self.distance_features["Mean distance"] + self.z_05 * std_rd)

        return self.compute_fractal_dimension(d_fd_cc)

    def compute_fractal_dimension_ce(self):
        """
        Compute the fractal dimension-CE.

        The fractal dimension-CE is based on the 95% confidence elipse.
        """

        std_ml = self.compute_std_ml()
        std_ap = self.compute_std_ap()

        try:
            d_fd_ce = np.sqrt(4 * self.F_05 * np.sqrt(np.square(np.square(std_ml)) + np.square(np.square(std_ap)) + 6 * np.square(
                std_ml) * np.square(std_ap) - 4 * np.square(np.cov(self.cop_x, self.cop_y)[0][1]) - (np.square(std_ml) + np.square(std_ap))))

            return self.compute_fractal_dimension(d_fd_ce)

        except Warning:
            return np.nan

    def compute_hybrid_features(self):
        """Compute all the hybrid features and store them in a dictionary."""

        features = {}
        features["Sway area"] = self.compute_sway_area()
        features["Mean frequency"] = self.compute_mean_frequency()
        features["Mean frequency-ML"] = self.compute_mean_frequency_ml()
        features["Mean frequency-AP"] = self.compute_mean_frequency_ap()
        features["Fractal dimension-CC"] = self.compute_fractal_dimension_cc()
        features["Fractal dimension-CE"] = self.compute_fractal_dimension_ce()

        return features

    def summary(self):
        """Print out a summary of the hybrid features to standard output."""

        for key, value in self.hybrid_features.items():
            print("{}: {}".format(key, value))


class TimeFeatures(HybridFeatures):
    """
    Class that merges all the time domain features.
    """

    def __init__(self, cop_x, cop_y):
        super(TimeFeatures, self).__init__(cop_x, cop_y)
        self.time_features = self.merge_time_features()

    def merge_time_features(self):
        """Merge all the time domain features together."""

        return {**self.distance_features, **self.area_features, **self.hybrid_features}

    def summary(self):
        """Print out a summary of the time features to standard output. """

        for key, value in self.time_features.items():
            print("{}: {}".format(key, value))
