import os
import h5py
import time

from datetime import timedelta
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5

from helper_functions import resample
from not_public.pydvs.preprocess_utils import *


"""

Here data is denoised with the conventional filter method used as comparison method in Section 4.3

"""

raw_data_path = "data/raw_DAS/"
denoised_data_path = "experiments/14_julius_filter/denoisedDAS"
files = os.listdir(raw_data_path)

print_count = 0

for i_file, file in enumerate(files):


    """ Load DAS File: """
    raw_file_path = os.path.join(raw_data_path, file)
    denoised_file_path = os.path.join(denoised_data_path, "denoised_julius_" + file)


    if not os.path.exists(denoised_file_path):

        print("Denoising File " + str(i_file), ": " + file)

        with h5py.File(raw_file_path, "r") as f:
            data = f["Acoustic"][:]
        headers = load_das_h5.load_headers_only(raw_file_path)

        """ Resample Data: """
        recording_fs = headers["fs"]
        data = resample(data, recording_fs / 400)
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928:
            data = data[:, ::6]
        else:
            data = data[:, ::3]
        headers["fs"] = 400
        headers["dx"] = 12
        data = data.astype("f")
        t_start = headers["t0"]


        """ Start Filtering """
        start = time.time()
        """ 1. Demean in time """
        time_mean = data.mean(axis=0).reshape(1, -1)
        data_demean = data - time_mean

        """ 2. Taper """
        data_tapered = taper(data_demean, max_percentage=0.05)

        """ 3. Bandpassfilter """ #OBACHT: Julius filtered zwischen 10 und 90 Hz
        data_filtered = bandpass_filter(data_tapered, freqmin=1, freqmax=120, fs=headers["fs"])

        """ 4. Automatic Gain Control """
        data_filtered_agc, tosum = AGC(data_filtered, headers["fs"] * 2)

        """ OPTIONAL 5. Demean in space """
        #data_filtered_time_mean = data_filtered_agc.mean(axis=1).reshape(-1, 1)
        #data_filtered_demean = data_filtered_agc - data_filtered_time_mean

        """ OPTIONAL 6. Energy normalization """
        #data_enorm = energy_norm(data_filtered_demean)
        end = time.time()

        """ Save data_filtered_agc as .h5 file"""
        #write_das_h5.write_block(data_filtered_agc, headers, denoised_file_path)

        """ Measuring Runtime: """
        if print_count < 30:
            print_count += 1
            dur = end - start
            dur_str = str(timedelta(seconds=dur))
            x = dur_str.split(':')
            with open('experiments/14_julius_filter/14_runtimes.txt', 'a') as w_file:
                w_file.write(str(print_count) + ": File " + str(file) + " took " + str(x[1]) + " minutes and " + str(
                    x[2]) + " seconds\n")

    else:
        print("File " + str(i_file), ": " + file, " already denoised. No Denoising is performed. ")