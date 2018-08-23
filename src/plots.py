# Built-in modules imports
from time_features import TimeFeatures


class TimeFeaturePlots(TimeFeatures):
    def __init__(self, cop_x, cop_y):
        super(TimeFeaturePlots, self).__init__(cop_x, cop_y)

    def plot_statokinesigram(data):
        pass

    def plot_stabilogram(data):
        pass

    def plot_power_sprectral_density(data):
        pass
