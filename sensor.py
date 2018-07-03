import btk
import json


class SensorDataReader(object):
    def __init__(self, filepath):
        self.config = self.load_config()
        self.acquisition_reader = btk.btkAcquisitionFileReader()
        self.acquisition_reader.SetFilename(filepath)
        self.acquisition_reader.Update()

    def set_reader_filename(self, filepath):
        self.acquisition_reader.SetFilename(filepath)
        self.acquisition_reader.Update()

    @staticmethod
    def load_config():
        with open("config/sensor.json") as cfg:
            config = json.load(cfg)
        return config

    def get_sensor_data(self, balance_board=False):
        reader = self.acquisition_reader
        acq = reader.GetOutput()

        if balance_board:
            data_points_labels = ["COP x y body weight", "BottomLeft Kg",
                                  "BottomRight Kg", "TopLeft Kg", "TopLeft Kg"]
            points = [acq.GetPoint(label)
                      for label in data_points_labels]
            values = [point.GetValues() for point in points]
        else:
            analog_labels = ["Fx1", "Fy1", "Mx1", "Fx2", "Fy2", "Mx2"]

            analogs = [acq.GetAnalog(label)
                       for label in analog_labels]
            values = [analog.GetValues() for analog in analogs]

        return values

    def get_frequency(self, point=False):
        reader = self.acquisition_reader
        acq = reader.GetOutput()

        if point:
            return acq.GetPointFrequency()
        else:
            return acq.GetAnalogFrequency()

    def format_sensor_data(self, data):
        pass
