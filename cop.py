import numpy as np
from utils import load_config
config = load_config()


def compute_cop_wbb_x(raw_data):
    """ Function to compute the x coordinate of the WBB center of pressure """

    # Wbb width (in mm)
    lx = config["width"] * 10

    # Wbb force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    cop_wbb_x = np.array(lx / 2 * ((TR + BR) - (TL + BL)) / (TR + BR + TL + BL))

    return cop_wbb_x


def compute_cop_wbb_y(raw_data):
    """ Function to compute the y coordinate of the WBB center of pressure """

    # Wbb length (in mm)
    ly = config["length"] * 10

    # Wbb force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    cop_wbb_y = np.array(ly / 2 * ((TL + TR) - (BR + BL)) / (TR + BR + TL + BL))

    return cop_wbb_y


def compute_cop_fp_x(raw_data):
    """ Function to compute the x coordinate of the force plate center of pressure """

    # Force plate heigth (in mm)
    dz = config["height"]

    # Force plate sensor values
    Fx = raw_data["Fx1"]
    My = raw_data["My1"]
    Fz = raw_data["Fz1"]

    cop_fp_x = (-My + dz * Fx) / Fz

    return cop_fp_x.flatten()


def compute_cop_fp_y(raw_data):
    """ Function to compute the y coordinate of the force plate center of pressure """

    # Force plate heigth (in mm)
    dz = config["height"]

    # Force plate sensor values
    Fy = raw_data["Fy1"]
    Mx = raw_data["Mx1"]
    Fz = raw_data["Fz1"]

    cop_fp_y = (Mx + dz * Fy) / Fz

    return cop_fp_y.flatten()
