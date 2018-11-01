# Third-party module imports
from time_features import TimeFeatures
from frequency_features import FrequencyFeatures


class DataProcessor:

    @staticmethod
    def compute_time_features(cop_x, cop_y):
        """Compute the time domain features."""

        time_domain_features = TimeFeatures(cop_x, cop_y)

        return time_domain_features.time_features

    @staticmethod
    def compute_frequency_features(cop_x, cop_y):
        """Compute the frequency domain features."""

        frequency_domain_features = FrequencyFeatures(cop_x, cop_y)

        return frequency_domain_features.frequency_features
