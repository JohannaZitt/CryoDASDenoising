# Filer of the paper "Denoising DAS data using an adaptive frequency-wavenumber filter"
# from Marius Isken et al. (2022)

"""

Miscellanous:
1. increasing the window size, the computation time also increases
2.


"""


from datetime import datetime, timedelta
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import h5py
from obspy import Stream, Trace, UTCDateTime, read
from lightguide.blast import Blast

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5
from helper_functions import resample


raw_data_path = "../data/raw_DAS/"
denoised_data_path = "experiments/13_isken_filter/denoisedDAS"
files = os.listdir(raw_data_path)

print_count = 0

for i_file, file in enumerate(files):


    """ Load DAS File: """
    raw_file_path = os.path.join(raw_data_path, file)
    denoised_file_path = os.path.join(denoised_data_path, "denoised_isken_" + file)


    if not os.path.exists(denoised_file_path):

        print("Denoising File " + str(i_file), ": " + file)

        with h5py.File(raw_file_path, "r") as f:
            data = f["Acoustic"][:]
        headers = load_das_h5.load_headers_only(raw_file_path)

        """ Resample Data: """
        recording_fs =  headers["fs"]
        data = resample(data, recording_fs / 400)
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928:
            data = data[:, ::6]
        else:
            data = data[:, ::3]
        headers["fs"] = 400
        headers["dx"] = 12
        data = data.astype("f")
        t_start = headers["t0"]

        # Generate .mseed file for using lightguide:
        st = Stream()
        for i in range(data.shape[1]):
            trace_data = np.ascontiguousarray(data[:, i])
            trace = Trace(data=trace_data)
            trace.stats.sampling_rate = headers["fs"]
            trace.stats.starttime = t_start
            trace.stats.station = str(i)
            st.append(trace)
        st.write("experiments/13_isken_filter/mini_seed_das.mseed", format="MSEED", encoding = 4)


        """ Denoise Data: """
        blast = Blast.from_miniseed("experiments/13_isken_filter/mini_seed_das.mseed")
        blast.bandpass(max_freq=120, min_freq=1)
        start = time.time()
        blast.afk_filter(exponent=0.8, window_size=32, overlap=7, normalize_power=False)
        denoised_data = blast.data
        end = time.time()

        #print(denoised_data.shape)
        #print(type(denoised_data))

        """ Save Denoised Data: """

        denoised_data = denoised_data.T
        print(denoised_data.shape)
        write_das_h5.write_block(denoised_data, headers, denoised_file_path)

        """ Measuring Runtime: """
        if print_count < 20:
            print_count += 1
            dur = end - start
            dur_str = str(timedelta(seconds=dur))
            x = dur_str.split(':')
            with open('experiments/13_isken_filter/13_runtimes.txt', 'a') as w_file:
                w_file.write(str(print_count) + ": File " + str(file) + " took " + str(x[1]) + " minutes and " + str(
                    x[2]) + " seconds\n")



    else:
        print("File " + str(i_file), ": " + file, " already denoised. No Denoising is performed. ")






#dur = end-start
#dur_str = str(timedelta(seconds=dur))
#x = dur_str.split(":")
#print("Exponent: " + str(exponent) + "; Window_Size: " + str(window_size) + "; Overlap: " + str(overlap) + "; Time: " + str(x[2]))





