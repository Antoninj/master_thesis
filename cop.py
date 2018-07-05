import json
import os
from utils import load_config

config = load_config()


def compute_cop_wbb_x(raw_data):
    """ Compute the x coordinate of the WBB center of pressure """

    # Wbb width (in mm)
    lx = config["width"] * 10

    # Force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    return lx / 2 * ((TR + BR) - (TL + BL)) / (TR + BR + TL + BL)


def compute_cop_wbb_y(raw_data):
    """ Compute the y coordinate of the WBB center of pressure """

    # Wbb length (in mm)
    ly = config["length"] * 10

    # Force sensor values
    TR = raw_data["TopRight Kg"][:, 0]
    BR = raw_data["BottomRight Kg"][:, 0]
    TL = raw_data["TopLeft Kg"][:, 0]
    BL = raw_data["BottomLeft Kg"][:, 0]

    return ly / 2 * ((TL + TR) - (BR + BL)) / (TR + BR + TL + BL)


def compute_cop_fp_x(raw_data):
    """ Compute the x coordinate of the force plate center of pressure """

    dz = config["height"]

    pass


def computer_cop_fp_y(raw_data):
    """ Compute the y coordinate of the force plate center of pressure """

    dz = config["height"]

    pass

