from time_features import DistanceFeatures
from utils import load_config

if __name__ == "__main__":

    # Load configuration file
    config = load_config()

    # WBB data test file
    filepath_wbb = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1_cop.json"

    distance_features = DistanceFeatures(filepath_wbb)

    rd_mean_distance = distance_features.compute_rd_mean_distance()
    ap_mean_distance = distance_features.compute_ap_mean_distance()
    ml_mean_distance = distance_features.compute_ml_mean_distance()

    print("Rd mean distance: {}".format(rd_mean_distance))
    print("AP mean distance: {}".format(ap_mean_distance))
    print("ML mean distance: {} \n".format(ml_mean_distance))

    rd_rms_distance = distance_features.compute_rd_rms_distance()
    ap_rms_distance = distance_features.compute_ap_rms_distance()
    ml_rms_distance = distance_features.compute_ml_rms_distance()

    print("Rd rms distance: {}".format(rd_rms_distance))
    print("AP rms distance: {}".format(ap_rms_distance))
    print("ML rms distance: {} \n".format(ml_rms_distance))

    rd_path_length = distance_features.compute_rd_path_length()
    ap_path_length = distance_features.compute_ap_path_length()
    ml_path_length = distance_features.compute_ml_path_length()

    print("Rd path length: {}".format(rd_path_length))
    print("AP path length: {}".format(ap_path_length))
    print("ML path length: {} \n".format(ml_path_length))
