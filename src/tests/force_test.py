# Third-party modules imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import btk

from context import *


# Force plate data test file
filepath_fp = config["test_files"]["fp_raw_data"]

acquisition_reader = btk.btkAcquisitionFileReader()
fp_extractor = btk.btkForcePlatformsExtractor()
wrench_filter = btk.btkGroundReactionWrenchFilter()


acquisition_reader.SetFilename(filepath_fp)
acquisition_reader.Update()
acq = acquisition_reader.GetOutput()

fp_extractor.SetInput(acq)
fp_collection = fp_extractor.GetOutput()

wrench_filter.SetInput(fp_collection)

wrench_collection = wrench_filter.GetOutput()

iterator = wrench_collection.Begin()
wrench_0 = iterator.value()

#forces = wrench_0.GetForce()

#print(forces)
