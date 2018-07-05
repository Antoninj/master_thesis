import argparse
import json
import os
from utils import load_config

if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("filepath", nargs=1,
                        help="Input image location", type=str)

    args = parser.parse_args()
