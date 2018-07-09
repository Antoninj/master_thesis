import numpy as np
from numpy import mean, sqrt, square, diff
from features import CopFeatures


class DistanceFeatures(CopFeatures):
    """                                       """

    def __init__(self, filepath):
        super(DistanceFeatures, self).__init__(filepath)
        self.distance_features = self.compute_distance_features()

    @staticmethod
    def compute_mean_distance(array):
        return array.mean()

    def compute_rd_mean_distance(self):
        return self.compute_mean_distance(self.cop_rd)

    def compute_ap_mean_distance(self):
        return self.compute_mean_distance(self.cop_x)

    def compute_ml_mean_distance(self):
        return self.compute_mean_distance(self.cop_y)

    @staticmethod
    def compute_rms_distance(array):
        rms_distance = sqrt(mean(square(array)))
        return rms_distance

    def compute_rd_rms_distance(self):
        return self.compute_rms_distance(self.cop_rd)

    def compute_ap_rms_distance(self):
        return self.compute_rms_distance(self.cop_x)

    def compute_ml_rms_distance(self):
        return self.compute_rms_distance(self.cop_y)

    @staticmethod
    def compute_path_length(array1, array2):
        distances_1 = diff(array1, axis=0)
        distances_2 = diff(array2, axis=0)
        path_length = np.sqrt((distances_1 ** 2) + (distances_2**2)).sum()

        return path_length

    def compute_rd_path_length(self):
        return self.compute_path_length(self.cop_x, self.cop_y)

    def compute_ap_path_length(self):
        distances = diff(self.cop_x, axis=0)
        path_length = distances.sum()

        return path_length

    def compute_ml_path_length(self):
        distances = np.diff(self.cop_y, axis=0)
        path_length = distances.sum()

        return path_length

    def compute_rd_mean_velocity(self):
        mean_velocity = (self.compute_rd_path_length()) / (self.cop_rd.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ap_mean_velocity(self):
        mean_velocity = (self.compute_ap_path_length()) / (self.cop_x.size / self.acquisition_frequency)
        return mean_velocity

    def compute_ml_mean_velocity(self):
        mean_velocity = (self.compute_ml_path_length()) / (self.cop_y.size / self.acquisition_frequency)
        return mean_velocity

    @staticmethod
    def compute_range(min_value, max_value):
        return max_value - min_value

    def compute_rd_range(self):
        return self.compute_range(self.cop_rd.min(), self.cop_rd.max())

    def compute_ap_range(self):
        return self.compute_range(self.cop_x.min(), self.cop_x.max())

    def compute_ml_range(self):
        return self.compute_range(self.cop_y.min(), self.cop_y.max())

    def compute_distance_features(self):
        features = {}
        features["Rd mean distance"] = self.compute_rd_mean_distance()
        features["AP mean distance"] = self.compute_ap_mean_distance()
        features["ML mean distance"] = self.compute_ml_mean_distance()
        features["Rd rms distance"] = self.compute_rd_rms_distance()
        features["AP rms distance"] = self.compute_ap_rms_distance()
        features["ML rms distance"] = self.compute_ml_rms_distance()
        features["Rd path length"] = self.compute_rd_path_length()
        features["AP path length"] = self.compute_ap_path_length()
        features["ML path length"] = self.compute_ml_path_length()
        features["Rd mean velocity"] = self.compute_rd_mean_velocity()
        features["AP mean velocity"] = self.compute_ap_mean_velocity()
        features["ML mean velocity"] = self.compute_ml_mean_velocity()
        features["Rd range"] = self.compute_rd_range()
        features["AP range"] = self.compute_ap_range()
        features["ML range"] = self.compute_ml_range()

        return features

    def summary(self):
        for key, value in self.distance_features.items():
            print("{}: {}".format(key, value))


class AreaFeatures(DistanceFeatures):
    """                                       """

    z_05 = 1.645
    F_05 = 3.0

    def __init__(self, filepath):
        super(AreaFeatures, self).__init__(filepath)
        self.area_features = self.compute_area_features()

    def compute_std_rd(self):
        std_rd = sqrt(square(self.distance_features["Rd rms distance"]) - square(self.distance_features["Rd mean distance"]))
        return std_rd

    def compute_std_ap(self):
        std_ap = sqrt(square(self.distance_features["AP rms distance"]) - square(self.distance_features["AP mean distance"]))
        return std_ap

    def compute_std_ml(self):
        std_ml = sqrt(square(self.distance_features["ML rms distance"]) - square(self.distance_features["ML mean distance"]))
        return std_ml

    def compute_confidence_circle_area(self):
        std_rd = self.compute_std_rd()
        area_cc = np.pi * square(self.distance_features["Rd mean distance"] + self.z_05 * std_rd)

        return area_cc

    def compute_confidence_elipse_area(self):
        std_ap = self.compute_std_ap()
        std_ml = self.compute_std_ml()
        area_ce = 2 * np.pi * self.F_05 * sqrt(square(std_ap) * square(std_ml) - np.cov(self.cop_x, self.cop_y)[0][1])

        return area_ce

    def compute_area_features(self):
        features = {}
        features["95% confidence circle area"] = self.compute_confidence_circle_area()
        features["95% confidence elipse area"] = self.compute_confidence_elipse_area()

        return features

    def summary(self):
        for key, value in self.area_features.items():
            print("{}: {}".format(key, value))


class HybridFeatures(AreaFeatures):
    """                                       """

    def __init__(self, filepath):
        super(HybridFeatures, self).__init__(filepath)
        self.hybrid_features = self.compute_hybrid_features()

    def compute_sway_area(self):
        sway_area = []
        T = (self.cop_rd.size / self.acquisition_frequency)
        for i in range(len(self.cop_rd) - 1):
            sway_area.append((self.cop_x[i + 1] * self.cop_y[i] - self.cop_x[i] * self.cop_y[i + 1]) / (2 * T))

        return np.array(sway_area).sum()

    def compute_mean_frequency(self):
        mean_frequency = (self.distance_features["Rd mean velocity"]) / (2 * np.pi * self.distance_features["Rd mean distance"])

        return mean_frequency

    def compute_mean_frequency_ap(self):
        mean_frequency_ap = (self.distance_features["AP mean velocity"]) / (4 * sqrt(2) * self.distance_features["AP mean distance"])

        return mean_frequency_ap

    def compute_mean_frequency_ml(self):
        mean_frequency_ml = (self.distance_features["ML mean velocity"]) / (4 * sqrt(2) * self.distance_features["ML mean distance"])

        return mean_frequency_ml

    def compute_fractal_dimension(self, d):
        N = self.cop_rd.size
        FD = (np.log(N)) / (np.log((N * d) / (self.distance_features["Rd path length"])))

        return FD

    def compute_fractal_dimension_cc(self):
        std_rd = self.compute_std_rd()
        d_fd_cc = 2 * (self.distance_features["Rd mean distance"] + self.z_05 * std_rd)

        return self.compute_fractal_dimension(d_fd_cc)

    def compute_fractal_dimension_ce(self):
        std_ap = self.compute_std_ap()
        std_ml = self.compute_std_ml()
        d_fd_ce = sqrt(8 * self.F_05 * sqrt(square(std_ap) * square(std_ml) - np.cov(self.cop_x, self.cop_y)[0][1]))

        return self.compute_fractal_dimension(d_fd_ce)

    def compute_hybrid_features(self):
        features = {}
        features["Sway area"] = self.compute_sway_area()
        features["Mean frequency"] = self.compute_mean_frequency()
        features["Mean frequency-AP"] = self.compute_mean_frequency_ap()
        features["Mean frequency-ML"] = self.compute_mean_frequency_ml()
        features["Fractal dimension-CC"] = self.compute_fractal_dimension_cc()
        features["Fractal dimension-CE"] = self.compute_fractal_dimension_ce()
        return features

    def summary(self):
        for key, value in self.hybrid_features.items():
            print("{}: {}".format(key, value))
