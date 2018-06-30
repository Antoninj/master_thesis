import sys
import btk


reader = btk.btkAcquisitionFileReader() # build a btk reader object
filepath = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/BalanceBoard/1.c3d"
reader.SetFilename(filepath) # set a filename to the reader
reader.Update()
acq = reader.GetOutput() # acq is the btk aquisition object

# Tests

point_freq = acq.GetPointFrequency() # get the point frequency
frames_number = acq.GetPointFrameNumber() # get the point number of frames

analog_frequency = acq.GetAnalogFrequency()
analog_frame_number =acq.GetAnalogFrameNumber()

print(analog_frequency, analog_frame_number)

#analog1 = acq.GetAnalog("Fx1")
point1 = acq.GetPoint("BottomLeft Kg")
point2 = acq.GetPoint("COP x y body weight")

values1 = point1.GetValues() # return a 1D Numpy array
values2 = point2.GetValues()

print(values1,values2)

