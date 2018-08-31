"""
Copyright 2016 Julien Audiffren


This code is related to the paper :
[Preprocessing of the Nintendo Wii Board Signal to Derive More
Accurate Descriptors of Statokinesigrams, Audiffren and Contal]

If you use this algorithm in your research work, please cite this paper.


Licence :
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later
 version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np


class SWARII:
    """
    Implementation of the Sliding Windows Weighted Averaged Interpolation method

    How To use :
        First instantiate the class with the desired parameters
        Then call resample on the desired signal

    """

    def __init__(self, window_size=1, desired_frequency=25):
        """
        Instantiate SWARII

        Parameters :
            desired_frequency : The frequency desired for the output signal,
                                after the resampling.
            window_size : The size of the sliding window, in seconds.
        """
        self.desired_frequency = desired_frequency
        self.window_size = window_size

    def resample(self, time, signal):
        """
        Apply the SWARII to resample a given signal.

        Input :
            time:   The time stamps of the the data point in the signal. A 1-d
                    array of shape n, where n is the number of points in the
                    signal. The unit is seconds.
            signal: The data points representing the signal. A k-d array of
                    shape (n,k), where n is the number of points in the signal,
                    and k is the dimension of the signal (e.g. 2 for a
                    statokinesigram).

        Output:
            resampled_time : The time stamps of the signal after the resampling
            resampled_signal : The resampled signal.
        """

        a_signal = np.array(signal)
        current_time = max(0., time[0])
        output_time = []
        output_signal = []

        while current_time < time[-1]:

            relevant_times = [t for t in range(len(time)) if abs(
                time[t] - current_time) < self.window_size * 0.5]

            assert len(relevant_times) > 0, "Trying to interpolate an empty window !"

            if len(relevant_times) == 1:
                value = a_signal[relevant_times[0]]

            else:
                value = 0
                weight = 0

                for i, t in enumerate(relevant_times):
                    if i == 0 or t == 0:
                        left_border = max(
                            time[0], (current_time - self.window_size * 0.5))

                    else:
                        left_border = 0.5 * (time[t] + time[t - 1])

                    if i == len(relevant_times) - 1:
                        right_border = min(
                            time[-1], current_time + self.window_size * 0.5)
                    else:
                        right_border = 0.5 * (time[t + 1] + time[t])

                    w = right_border - left_border

                    value += a_signal[t] * w
                    weight += w

                value /= weight
            output_time.append(current_time)
            output_signal.append(value)
            current_time += 1. / self.desired_frequency

        return np.array(output_time), np.array(output_signal)
