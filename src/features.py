# Built-in modules imports
from utils import load_config

# Third-party module imports
import numpy as np
import json

config = load_config()


class CopFeatures(object):
    """
    Base class for COP based features computations

    This class doesn't implement any feature computations and is simply responsible for parsing the COP data that will be used in COP based features computations in both the time and frequency domains.
    """

    acquisition_frequency = config["preprocessing_parameters"]["acquisition_frequency"]

    def __init__(self, cop_x, cop_y):
        self.cop_x = cop_x
        self.cop_y = cop_y
        self.cop_rd = self.compute_rd(self.cop_x, self.cop_x)

    @classmethod
    def from_file(cls, filepath):
        cop_x, cop_y = cls.parse_cop_data(filepath)

        return cls(cop_x, cop_y)

    @staticmethod
    def parse_cop_data(filepath):
        """ Function to parse cop data from an input file """

        with open(filepath) as json_data:
            cop_data = json.load(json_data)

        cop_x = np.array(cop_data["COP_x"])
        cop_y = np.array(cop_data["COP_y"])

        return cop_x, cop_y

    @staticmethod
    def compute_rd(cop_x, cop_y):
        """ Function to compute the resultant distance vector from the x and y COP coordinates """

        return np.array([np.sqrt(x**2 + y**2) for x, y in zip(cop_x, cop_y)])
