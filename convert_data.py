import sys
import btk


reader = btk.btkAcquisitionFileReader() # build a btk reader object
filepath = "/Users/Antonin/Documents/VUB/semester 4/thesis/code/BalanceBoard_Static/Sujet1/Session1/Vicon/1.c3d"
reader.SetFilename(filepath) # set a filename to the reader
reader.Update()
acq = reader.GetOutput() # acq is the btk aquisition object

analog1 = acq.GetAnalog("Fx1") # attribute a btk-analog object to the measurement
values = analog1.GetValues() # return a 1D Numpy array

point_freq = acq.GetPointFrequency() # give the point frequency
frames_number = acq.GetPointFrameNumber() # give the number of frames

print(point_freq, frames_number)

print(values)
