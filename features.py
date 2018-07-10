import numpy as np
import json
from utils import load_config

config = load_config("preprocess")


class CopFeatures(object):
    """
    Base class for COP based features computations

    This class doesn't implement any feature computations and is simply responsible for parsing the COP data that will be used in COP based features computations in both the time and frequency domains.
    """

    acquisition_frequency = config["acquisition_frequency"]

    def __init__(self, filepath):
        super(CopFeatures, self).__init__()
        self.parse_cop_data(filepath)
        self.cop_rd = self.compute_rd(self.cop_x, self.cop_x)

    def parse_cop_data(self, filepath):
        """ Function to parse cop data from an input file """

        with open(filepath) as json_data:
            cop_data = json.load(json_data)

        self.cop_x = np.array(cop_data["COP x"])
        self.cop_y = np.array(cop_data["COP y"])

    @staticmethod
    def compute_rd(cop_x, cop_y):
        """ Function to compute the resultant distance vector from the x and y COP coordinates """

        return np.array([np.sqrt(x**2 + y**2) for x, y in zip(cop_x, cop_y)])
