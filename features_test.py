from time_features import DistanceFeatures, AreaFeatures, HybridFeatures
from utils import load_config

if __name__ == "__main__":

    # Load configuration file
    config = load_config("test")

    # WBB data test file
    wbb_cop_data = config["wbb cop data test file"]

    # Time features tests

    distance_features = DistanceFeatures(wbb_cop_data)

    distance_features.summary()

    area_features = AreaFeatures(wbb_cop_data)

    area_features.summary()

    hybrid_features = HybridFeatures(wbb_cop_data)

    hybrid_features.summary()

    # Frequency features tests

