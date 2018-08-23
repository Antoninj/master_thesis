# Built-in modules imports
from utils import load_config

# Third-party module imports
import numpy as np
import pandas as pd
import warnings
import logging

# Set numpy error level to warning
np.seterr(all='warn')

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('error')

logging.captureWarnings(True)
config = load_config()


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)

    return arr[prev]


def compute_cop_wbb_x(data):
    """Compute the x coordinate of the WBB center of pressure (ML direction)."""

    # Wbb width (in mm)
    lx = config["wbb_parameters"]["width"]

    # Wbb force sensor values
    TR = data["TopRight Kg"]
    BR = data["BottomRight Kg"]
    TL = data["TopLeft Kg"]
    BL = data["BottomLeft Kg"]

    TR = pd.DataFrame(TR)[0].replace(to_replace=0, value=1).values
    BR = pd.DataFrame(BR)[0].replace(to_replace=0, value=1).values
    TL = pd.DataFrame(TL)[0].replace(to_replace=0, value=1).values
    BL = pd.DataFrame(BL)[0].replace(to_replace=0, value=1).values

    try:
        cop_wbb_x = np.array((lx / 2) * ((TR + BR) - (TL + BL)) / (TR + BR + TL + BL))

        return cop_wbb_x

    except Warning:
        return np.ones_like(TR)


def compute_cop_wbb_y(data):
    """Compute the y coordinate of the WBB center of pressure (AP direction)."""

    # Wbb length (in mm)
    ly = config["wbb_parameters"]["length"]
    # Wbb force sensor values
    TR = data["TopRight Kg"]
    BR = data["BottomRight Kg"]
    TL = data["TopLeft Kg"]
    BL = data["BottomLeft Kg"]

    TR = pd.DataFrame(TR)[0].replace(to_replace=0, value=1).values
    BR = pd.DataFrame(BR)[0].replace(to_replace=0, value=1).values
    TL = pd.DataFrame(TL)[0].replace(to_replace=0, value=1).values
    BL = pd.DataFrame(BL)[0].replace(to_replace=0, value=1).values

    try:
        cop_wbb_y = np.array((ly / 2) * ((TL + TR) - (BR + BL)) / (TR + BR + TL + BL))

        return cop_wbb_y

    except Warning:
        return np.ones_like(TR)


def compute_cop_fp_x(data, debug=False):
    """Compute the x coordinate of the force plate center of pressure."""

    # Force plate height (in mm)
    dz = config["wbb_parameters"]["height"]

    # Force plate sensor values
    Fx = data["Fx1"]
    My = data["My1"]
    Fz = data["Fz1"]

    Fx = pd.DataFrame(Fx)[0].replace(to_replace=0, value=1).values
    My = pd.DataFrame(My)[0].replace(to_replace=0, value=1).values
    Fz = pd.DataFrame(Fz)[0].replace(to_replace=0, value=1).values

    cop_fp_x = -(My + dz * Fx) / (Fz)

    return cop_fp_x


def compute_cop_fp_y(data):
    """Compute the y coordinate of the force plate center of pressure."""

    # Force plate height (in mm)
    dz = config["wbb_parameters"]["height"]

    # Force plate sensor values
    Fy = data["Fy1"]
    Mx = data["Mx1"]
    Fz = data["Fz1"]

    Fy = pd.DataFrame(Fy)[0].replace(to_replace=0, value=1).values
    Mx = pd.DataFrame(Mx)[0].replace(to_replace=0, value=1).values
    Fz = pd.DataFrame(Fz)[0].replace(to_replace=0, value=1).values

    cop_fp_y = (Mx + dz * Fy) / (Fz)

    return cop_fp_y
