import gc
import h5py
import numpy as np
import os
import time
from datetime import timedelta
from tensorflow import keras

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5

from pyDASDL.Utils import patch, patch_inv
from pyDASDL.cwt_2d import cwt_2d
from pyDASDL.mf import mf

from helper_functions import butter_bandpass_filter


def reshape(data, ratio):

    data = data.T

    # Downsample data in space and time:
    n_ch = data.shape[0]
    if n_ch == 4800 or n_ch == 4864 or n_ch == 4928:
        data = data[::6, :]
    else:
        data = data[::3, :]

    # time axis
    res = np.zeros((data.shape[0], int(data.shape[1] / ratio)))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res


files_path = "data/raw_DAS"
files = os.listdir(files_path)

"""  Load model  """
model = keras.models.load_model("experiments/15_DASDL/DASDL_models/DAS_PATCH_rohnegletscher_sample_82.h5")
#model.summary()

""" Parameters """
w1 = 48
w2 = 48
s1z = 8
s2z = 8
lowcut = 0.001
highcut = 120
order = 4
print_count = 0

for i_file, file in enumerate(files):

    denoised_file_path = "experiments/15_DASDL/denoisedDAS/denoised_DASDL_" + file + ".h5"

    if not os.path.exists(denoised_file_path):

        print("Denoising File ", str(i_file), ": " + file)

        """  Load DAS data """
        file_path = os.path.join(files_path, file)
        with h5py.File(file_path, "r") as f:
            data = f["Acoustic"][:]
        headers = load_das_h5.load_headers_only(file_path)
        fs_orig=headers["fs"]
        fs = 400
        data = reshape(data, fs_orig / fs)
        headers["fs"] = fs


        """ Preprocess Data"""
        n_ch, n_time = data.shape
        print(data.shape)
        data_bp = np.zeros_like(data)  # To store the filtered output
        for ch in range(n_ch):
            data_bp[ch, :] = butter_bandpass_filter(data[ch, :], lowcut, highcut, fs, order)
        #data_bpmf = mf(data_bp, 5, 1, 1) # median filter



        """ Compute Freq. Domain Data """
        ss_log = np.arange(1, 10.5, 0.5)
        coeffs_nsq, wav_norm_nsq = cwt_2d(data_bp, ss_log, 'mexh')  # (data, scale ,wavelength args) mexican hat
        data_cwt = np.abs(coeffs_nsq[:, :,- 1])
        ma = np.max(np.abs(data_cwt))
        data_cwt_norm = data_cwt / ma


        """ Patching the CWT SCALE """
        dataInputF = patch(data_cwt_norm, w1, w2, s1z, s2z)
        dataInputF2 = np.reshape(dataInputF, (dataInputF.shape[0], w1 * w2, 1))

        """ Patching the Band-pass Filtered Data """
        dataInputT = patch(data_bp, w1, w2, s1z, s2z)
        dataInputT2 = np.reshape(dataInputF, (dataInputF.shape[0], w1 * w2, 1))

        """ Denoising Data """
        start = time.time()
        out = model.predict([dataInputT2, dataInputF2], batch_size=64) # Time and Frequency domain
        end = time.time()

        """ Reshape """
        out = np.reshape(out, (out.shape[0], w1 * w2))
        outA = np.transpose(out)
        n1, n2 = np.shape(data_bp)
        outB = patch_inv(outA, n1, n2, w1, w2, s1z, s2z)
        denoised_data = np.array(outB)
        denoised_data = denoised_data.T

        """ Save Data """
        write_das_h5.write_block(denoised_data, headers, denoised_file_path)

        """ Measuring Runtime: """
        if print_count < 50:
            print_count += 1
            dur = end - start
            dur_str = str(timedelta(seconds=dur))
            x = dur_str.split(':')
            with open('experiments/15_DASDL/15_runtimes.txt', 'a') as w_file:
                w_file.write(str(print_count) + ": File " + str(file) + " took " + str(x[1]) + " minutes and " + str(
                    x[2]) + " seconds\n")

    else:
        print("File " + str(i_file), ": " + file, " already denoised. No Denoising is performed.")

    gc.collect()

