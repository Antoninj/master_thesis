# Third-party module imports
import json
import numpy as np
import os
import logging.config
import sys
#from matplotlib import pyplot as plt, gridspec


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


def save_as_json(data, filepath, folder_to_replace, destination_folder, name_extension):
    """Save results to json format."""

    filename = build_filename(filepath, folder_to_replace, destination_folder, name_extension)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder, sort_keys=False, indent=4, ensure_ascii=False)


def build_filename(input_file, folder_to_replace, destination_folder, name_extension):
    """Build a custom destination filepath from the input file."""

    base_name = os.path.splitext(input_file)[0]
    filename = base_name.replace(folder_to_replace, destination_folder) + "{}".format(name_extension)
    dir_name = os.path.dirname(filename)
    check_folder(dir_name)

    return filename


def check_folder(folder_name):
    """Check if a folder exists, and if not, create it."""

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def check_folders(folders):
    for folder in folders:
        check_folder(folder)


def get_path_to_all_files(folder_name):
    """Recursively get all filepaths from a directory tree."""

    exceptions_extensions = [".DS_Store", ".xls", ".tif"]
    try:
        filepaths = []
        for dirname, dirnames, filenames in os.walk(folder_name):
            for filename in filenames:
                if not any(ext in filename for ext in exceptions_extensions):
                    filepaths.append(os.path.join(dirname, filename))
        return filepaths

    except IOError as err:
        logger.critical(err, exc_info=True)
        sys.exit()


def separate_files(files):
    """Separate WBB and force plate data."""

    wbb_files = [file for file in files if "FP" not in file]
    fp_files = [file for file in files if "FP" in file]

    wbb_files_modified = [filename.replace("BB", "FP") for filename in wbb_files]
    fp_files_modified = [filename.replace("FP", "BB") for filename in fp_files]

    fp_files_curated = [file for file in fp_files if file in wbb_files_modified]
    wbb_files_curated = [file for file in wbb_files if file in fp_files_modified]
    identical_order_test = [i for i, j in zip(fp_files_curated, wbb_files_curated) if i == j.replace("BB", "FP")]
    if len(identical_order_test) == len(fp_files_curated):
        logger.error("The data to be analysed is not correctly ordered")

    return wbb_files_curated, fp_files_curated


def setup_logging(default_level=logging.INFO):
    """Setup the logging module configuration from configuration file."""

    config = load_config(filename="logging")
    if config:
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def compute_rd(cop_x, cop_y):
    """Compute the resultant distance vector from the x and y COP coordinates."""

    return np.array([np.sqrt(x**2 + y**2) for x, y in zip(cop_x, cop_y)])


def plot_stabilograms(preprocessed_cop_data, device_name, acq_frequency, filepath=None):
    """"Plot and save stabilograms from COP data."""

    cop_x = preprocessed_cop_data["COP_x"]
    cop_y = preprocessed_cop_data["COP_y"]
    cop_rd = compute_rd(cop_x, cop_y)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    index = [i / acq_frequency for i in range(len(cop_x))]
    fig.suptitle("{} plots".format(device_name), fontsize=16)

    axs[0][0].plot(index, cop_x)
    axs[0][0].set_xlabel('Time (seconds)')
    axs[0][0].set_ylabel('ML distance (mm)')
    axs[0][1].plot(index, cop_y)
    axs[0][1].set_xlabel('Time (seconds)')
    axs[0][1].set_ylabel('AP distance (mm)')
    axs[1][0].plot(index, cop_rd)
    axs[1][0].set_xlabel('Time (seconds)')
    axs[1][0].set_ylabel('Resultant distance (mm)')
    axs[1][1].plot(cop_x, cop_y)
    axs[1][1].set_xlabel('ML distance (mm)')
    axs[1][1].set_ylabel('AP distance (mm)')

    # Save the plots
    if filepath:
        config = load_config()
        swarii_window = config["preprocessing_parameters"]["swarii_window_size"]
        fig_name = build_filename(filepath, folder_to_replace="BalanceBoard/Repro", destination_folder="results/cop_plots", name_extension="_SWARII_{}.png".format(swarii_window))
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)

def plot_spectral_densities(frequencies, spectrums, filepath=None):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(15, 10))

    ax = plt.subplot(gs[0, 0])
    plt.plot(frequencies[0], spectrums[0])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectrum density in AP direction [mm**2/Hz]')

    ax = plt.subplot(gs[0, 1])
    plt.plot(frequencies[1], spectrums[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power spectrum density in ML direction [mm**2/Hz]')

    ax = plt.subplot(gs[1, :])
    plt.plot(frequencies[2], spectrums[2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Resultant distance Power spectrum density[mm**2/Hz]')

    # Save the plots
    if filepath:
        fig_name = build_filename(filepath, folder_to_replace="cop_data", destination_folder="spectrum_plots", name_extension="_spectrum.png")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)

