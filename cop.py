import numpy as np
import pandas as pd
from utils import load_config
import warnings

config = load_config()
# Set numpy error level to warning
np.seterr(all='warn')

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('error')


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)

    return arr[prev]


def compute_cop_wbb_x(raw_data):
    """ Function to compute the x coordinate of the WBB center of pressure """

    # Wbb width (in mm)
    lx = config["wbb_parameters"]["width"] * 10

    # Wbb force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    TR = pd.DataFrame(TR)[0].replace(to_replace=0, value=1).values
    BR = pd.DataFrame(BR)[0].replace(to_replace=0, value=1).values
    TL = pd.DataFrame(TL)[0].replace(to_replace=0, value=1).values
    BL = pd.DataFrame(BL)[0].replace(to_replace=0, value=1).values

    try:
        cop_wbb_x = np.array((lx / 2) * ((TR + BR) - (TL + BL)) / (TR + BR + TL + BL))

        return cop_wbb_x

    except Warning:
        return np.ones_like(TR)


def compute_cop_wbb_y(raw_data):
    """ Function to compute the y coordinate of the WBB center of pressure """

    # Wbb length (in mm)
    ly = config["wbb_parameters"]["length"] * 10

    # Wbb force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    TR = pd.DataFrame(TR)[0].replace(to_replace=0, value=1).values
    BR = pd.DataFrame(BR)[0].replace(to_replace=0, value=1).values
    TL = pd.DataFrame(TL)[0].replace(to_replace=0, value=1).values
    BL = pd.DataFrame(BL)[0].replace(to_replace=0, value=1).values

    try:
        cop_wbb_y = np.array((ly / 2) * ((TL + TR) - (BR + BL)) / (TR + BR + TL + BL))

        return cop_wbb_y

    except Warning:
        return np.ones_like(TR)


def compute_cop_fp_x(raw_data, debug=False):
    """ Function to compute the x coordinate of the force plate center of pressure """

    # Force plate heigth (in mm)
    dz = config["wbb_parameters"]["height"]

    # Force plate sensor values
    Fx = raw_data["Fx1"].flatten()
    My = raw_data["My1"].flatten()
    Fz = raw_data["Fz1"].flatten()

    Fx = pd.DataFrame(Fx)[0].replace(to_replace=0, value=1).values
    My = pd.DataFrame(My)[0].replace(to_replace=0, value=1).values
    Fz = pd.DataFrame(Fz)[0].replace(to_replace=0, value=1).values

    if debug:
        print(Fx)
        #print(np.where(FZ == 0)[0])
        # print(FZ[0:600])

    cop_fp_x = -(My + dz * Fx) / (Fz)

    return cop_fp_x


def compute_cop_fp_y(raw_data):
    """ Function to compute the y coordinate of the force plate center of pressure """

    # Force plate heigth (in mm)
    dz = config["wbb_parameters"]["height"]

    # Force plate sensor values
    Fy = raw_data["Fy1"].flatten()
    Mx = raw_data["Mx1"].flatten()
    Fz = raw_data["Fz1"].flatten()

    Fy = pd.DataFrame(Fy)[0].replace(to_replace=0, value=1).values
    Mx = pd.DataFrame(Mx)[0].replace(to_replace=0, value=1).values
    Fz = pd.DataFrame(Fz)[0].replace(to_replace=0, value=1).values

    cop_fp_y = (Mx - dz * Fy) / (Fz)

    return cop_fp_y
