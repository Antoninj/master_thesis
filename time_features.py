import numpy as np
from numpy import mean, sqrt, square, diff
import json
# from utils import load_config


class TimeFeatures(object):
    """                             """

    def __init__(self, cop_data):
        self.parse_cop_data(cop_data)

    def parse_cop_data(self, filepath):
        with open(filepath) as json_data:
            cop_data = json.load(json_data)

        self.cop_x = np.array(cop_data["COP x"])
        self.cop_y = np.array(cop_data["COP y"])
        self.cop_rd = self.compute_rd(self.cop_x, self.cop_x)

    @staticmethod
    def compute_rd(cop_x, cop_y):
        return np.array([sqrt(x**2 + y**2) for x, y in zip(cop_x, cop_y)])


class DistanceFeatures(TimeFeatures):
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
        return sqrt(mean(square(array)))

    def compute_rd_rms_distance(self):
        return self.compute_rms_distance(self.cop_rd)

    def compute_ap_rms_distance(self):
        return self.compute_rms_distance(self.cop_x)

    def compute_ml_rms_distance(self):
        return self.compute_rms_distance(self.cop_y)

    @staticmethod
    def compute_path_length(array1, array2):
        distances_1 = np.diff(array1, axis=0)
        distances_2 = np.diff(array2, axis=0)
        path_length = np.sqrt((distances_1 ** 2) + (distances_2**2)).sum()

        return path_length

    def compute_rd_path_length(self):
        return self.compute_path_length(self.cop_x, self.cop_y)

    def compute_ap_path_length(self):
        distances = np.diff(self.cop_x, axis=0)
        path_length = distances.sum()

        return path_length

    def compute_ml_path_length(self):
        distances = np.diff(self.cop_y, axis=0)
        path_length = distances.sum()

        return path_length
