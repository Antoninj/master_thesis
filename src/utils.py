# Third-party module imports
import json
import numpy as np
import os
import logging.config
import sys

logger = logging.getLogger("utils")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_config(filename="config"):
    """Load a configuration file."""

    path = os.path.abspath(os.path.dirname(__file__))
    config_path = "{}/config/{}.json".format(path, filename)

    try:
        with open(config_path) as file:
            config = json.load(file)
        return config

    except IOError as err:
        logger.critical(err, exc_info=True)
        sys.exit()


def save_as_json(data, filepath, destination_folder, name_extension):
    """Save results data to json format."""

    filename = build_filename(filepath, destination_folder, name_extension)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder, sort_keys=False, indent=4, ensure_ascii=False)


def build_filename(filepath, destination_folder, name_extension):
    """Build a custom destination filepath."""

    base_name = os.path.splitext(filepath)[0]
    replacement_string = "results/{}".format(destination_folder)
    filename = base_name.replace("BalanceBoard_Static", replacement_string) + "_{}.json".format(name_extension)
    dir_name = os.path.dirname(filename)
    check_folder(dir_name)

    return filename


def check_folder(folder_name):
    """Check if a folder exists, and if not, create it."""

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_path_to_all_files(folder_name):
    """Recursively get all filepaths from a directory tree."""

    try:
        filepaths = []
        for dirname, dirnames, filenames in os.walk(folder_name):
            for filename in filenames:
                if '.DS_Store' not in filename:
                    filepaths.append(os.path.join(dirname, filename))
        return filepaths

    except IOError as err:
        logger.critical(err, exc_info=True)
        sys.exit()


def clean_files(files):
    """Separate WBB and force plate data."""

    wbb_files = [file for file in files if "Vicon" not in file and "cop" not in file]
    fp_files = [file for file in files if "Vicon" in file and "cop" not in file]

    wbb_files_modified = [filename.replace("BalanceBoard", "Vicon") for filename in wbb_files]
    fp_files_modified = [filename.replace("Vicon", "BalanceBoard") for filename in fp_files]

    fp_files_curated = [file for file in fp_files if file in wbb_files_modified]
    wbb_files_curated = [file for file in wbb_files if file in fp_files_modified]

    return wbb_files_curated, fp_files_curated


def setup_logging(default_level=logging.INFO):
    """Setup the logging module configuration from configuration file."""

    config = load_config(filename="logging")
    if config is not None:
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
