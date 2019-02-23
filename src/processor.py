# Third-party module imports
from time_features import TimeFeatures
from frequency_features import FrequencyFeatures


class DataProcessor:

    @staticmethod
    def compute_time_features(cop_data_file):
        """Compute the time domain features."""

        time_domain_features = TimeFeatures.from_file(cop_data_file)

        return time_domain_features.time_features

    @staticmethod
    def compute_frequency_features(cop_data_file):
        """Compute the time domain features."""

        frequency_domain_features = FrequencyFeatures.from_file(cop_data_file)

        return frequency_domain_features
