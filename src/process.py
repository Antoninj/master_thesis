# Built-in modules imports
from utils import load_config

# Third-party module imports
import numpy as np
import pandas as pd
import warnings
import logging
from time_features import TimeFeatures
from frequency_features import FrequencyFeatures

# Set numpy error level to warning
np.seterr(all='warn')

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('error')

logging.captureWarnings(True)
config = load_config()


class DataProcessor:

    def compute_cop_positions(self, preprocessed_data, balance_board=False):
        """Compute the COP positions in the AP and ML directions."""

        cop_data = {}
        try:
            if balance_board:
                cop_data["COP_x"] = preprocessed_data["COP_x"]
                cop_data["COP_y"] = preprocessed_data["COP_y"]
            else:
                cop_data["COP_x"] = self.compute_cop_fp_x(preprocessed_data)
                cop_data["COP_y"] = self.compute_cop_fp_y(preprocessed_data)

            return cop_data

        except Exception:
            raise

    @staticmethod
    def compute_time_features(cop_x, cop_y):
        """Compute the time domain features."""

        time_domain_features = TimeFeatures(cop_x, cop_y)

        return time_domain_features.time_features

    @staticmethod
    def compute_frequency_features(cop_x, cop_y):
        """Compute the frequency domain features."""

        frequency_domain_features = FrequencyFeatures(cop_x, cop_y)

        return frequency_domain_features.frequency_features

    @staticmethod
    def compute_cop_fp_x(data):
        """Compute the x coordinate of the force plate center of pressure (ML direction)."""

        # Force plate height (in mm)
        dz = config["wbb_parameters"]["height"]

        # Force plate sensor values
        Fx1 = data["Fx1"]
        My1 = data["My1"]
        Fz1 = data["Fz1"]

        Fx1 = pd.DataFrame(Fx1)[0].replace(to_replace=0, value=1).values
        My1 = pd.DataFrame(My1)[0].replace(to_replace=0, value=1).values
        Fz1 = pd.DataFrame(Fz1)[0].replace(to_replace=0, value=1).values

        cop_fp_x = -(My1 + dz * Fx1) / (Fz1)

        return cop_fp_x

    @staticmethod
    def compute_cop_fp_y(data):
        """Compute the y coordinate of the force plate center of pressure (AP direction)."""

        # Force plate height (in mm)
        dz = config["wbb_parameters"]["height"]

        # Force plate sensor values
        Fy1 = data["Fy1"]
        Mx1 = data["Mx1"]
        Fz1 = data["Fz1"]

        Fy1 = pd.DataFrame(Fy1)[0].replace(to_replace=0, value=1).values
        Mx1 = pd.DataFrame(Mx1)[0].replace(to_replace=0, value=1).values
        Fz1 = pd.DataFrame(Fz1)[0].replace(to_replace=0, value=1).values

        cop_fp_y = (Mx1 - dz * Fy1) / (Fz1)

        return cop_fp_y
