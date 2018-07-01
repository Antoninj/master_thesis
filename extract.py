import btk
import os
import json


class DataExtractor(object):
    def __init__(self):
        self.config = self.load_config()
        self.acquisition_reader = btk.btkAcquisitionFileReader()

    @staticmethod
    def load_config():
        with open("config/config.json") as cfg:
            config = json.load(cfg)
        return config

    def get_force_plate_data(self, filepath):
        reader = self.acquisition_reader()
        reader.SetFilename(filepath)
        reader.Update()
        acq = reader.GetOutput()

        analog_labels = ["Fx1", "Fy1", "Mx1", "Fx2", "Fy2", "Mx2"]
        analogs = [acq.GetAnalog(label)
                   for label in analog_labels]
        analog_values = [analog.GetValues() for analog in analogs]

        return analog_values

    def get_wii_balance_data(self, filepath):
        reader = self.acquisition_reader()
        reader.SetFilename(filepath)
        reader.Update()
        acq = reader.GetOutput()

        data_points_labels = ["COP x y body weight", "BottomLeft Kg",
                              "BottomRight Kg", "TopLeft Kg", "TopLeft Kg"]
        points = [acq.GetPoint(label)
                  for label in data_points_labels]
        point_values = [point.GetValues() for point in points]

        return point_values
