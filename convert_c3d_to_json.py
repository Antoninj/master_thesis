import btk
#import argparse
#import os
import json

if __name__ == "__main__":

    # Load configuration file
    with open("config/config.json") as cfg:
        config = json.load(cfg)

    # Tests

    reader = btk.btkAcquisitionFileReader()
    filepath = "filepath"
    reader.SetFilename(filepath)
    reader.Update()
    acq = reader.GetOutput()

    point_freq = acq.GetPointFrequency()  # get the point frequency
    frames_number = acq.GetPointFrameNumber()  # get the point number of frames

    analog_frequency = acq.GetAnalogFrequency()
    analog_frame_number = acq.GetAnalogFrameNumber()

    print(analog_frequency, analog_frame_number)

    # analog1 = acq.GetAnalog("Fx1")
    point1 = acq.GetPoint("BottomLeft Kg")
    point2 = acq.GetPoint("COP x y body weight")

    values1 = point1.GetValues()  # return a 1D Numpy array
    values2 = point2.GetValues()

    print(values1, values2)
