import gc
from sys import prefix

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
from pyDASDL.fk import fkdip
from pyDASDL.bp import bandpass

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


def split_array(arr, n=9):
    side = int(np.sqrt(n))
    return [
        arr[i // side * arr.shape[0] // side:(i // side + 1) * arr.shape[0] // side,
        (i % side) * arr.shape[1] // side:(i % side + 1) * arr.shape[1] // side]
        for i in range(n)
    ]

def reassemble_blocks(blocks):
    side = int(np.sqrt(len(blocks)))
    rows = []
    for i in range(side):
        row_blocks = blocks[i*side:(i+1)*side]
        row = np.concatenate(row_blocks, axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def denoise_data_DASDL(data, model):

    w1 = 48
    w2 = 48
    s1z = 8
    s2z = 8

    """ Preprocess Data"""
    data = data.T
    n_ch, n_time = data.shape
    print("n_ch, n_time = ", data.shape)
    data_bp = bandpass(data, 0.001, 0.001, 120, 6, 6, 0, 0)
    #data_bpmf = mf(data_bp, 5, 1, 1) # median filter

    """ Compute Freq. Domain Data """
    ss_log = np.arange(1, 10.5, 0.5)
    coeffs_nsq, wav_norm_nsq = cwt_2d(data_bp, ss_log, 'mexh')  # (data, scale ,wavelength args) mexican hat
    data_cwt = np.abs(coeffs_nsq[:, :, - 1])
    #ma = np.max(np.abs(data_cwt))
    #data_cwt_norm = data_cwt / ma

    """ Patching the CWT SCALE """
    cwt_scale_patch = patch(data_cwt, w1, w2, s1z, s2z)
    cwt_scale_input = np.reshape(cwt_scale_patch, (cwt_scale_patch.shape[0], w1 * w2, 1))

    print(w1, w2, s1z, s2z)

    """ Patching the Band-pass Filtered Data """
    band_pass_patch = patch(data_bp, w1, w2, s1z, s2z)
    band_pass_input = np.reshape(band_pass_patch, (band_pass_patch.shape[0], w1 * w2, 1))

    """ Denoising Data """
    #start = time.time()
    out = model.predict([band_pass_input, cwt_scale_input], batch_size=64)  # Time and Frequency domain
    #end = time.time()

    """ Reshape """
    out = np.reshape(out, (out.shape[0], w1 * w2))
    outA = np.transpose(out)
    n1, n2 = np.shape(data_bp)
    outB = patch_inv(outA, n1, n2, w1, w2, s1z, s2z)
    denoised_data = np.array(outB)

    # here you could ad dip filter

    denoised_data = denoised_data.T


    return  denoised_data


def save_split_arrays(split_data, prefix="block"):
    for i, block in enumerate(split_data):
        filename = f"{prefix}_{i}.npy"
        np.save(filename, block)
        print(f"Saved: {filename}")



files_path = "data/raw_DAS_image"
files = os.listdir(files_path)

"""  Load model  """
model = keras.models.load_model("experiments/15_DASDL/DASDL_models/DAS_PATCH_Johanna.h5")
#model.summary()

""" Parameters """
lowcut = 0.001
highcut = 120
order = 4
print_count = 0

for i_file, file in enumerate(files):

    denoised_file_path = "experiments/15_DASDL/test/denoised_DASDL_" + file + ".h5"

    if not os.path.exists(denoised_file_path):

        print("Denoising File ", str(i_file), ": " + file)

        """  Load DAS data """
        file_path = os.path.join(files_path, file)
        with h5py.File(file_path, "r") as f:
            data = f["Acoustic"][:]
        headers = load_das_h5.load_headers_only(file_path)
        print("raw data shape: ", data.shape)
        fs=headers["fs"]


        # ensure channel spacing of 4 m
        n_ch_orig = data.shape[0]
        if n_ch_orig == 4800 or n_ch_orig == 4864 or n_ch_orig == 4928:
            data = data[::2, :]

        # split data for denoising
        split_data = split_array(data)

        # save blox
        save_split_arrays(split_data, prefix="experiments/15_DASDL/cwt_data/numpy_array/" + file)

        #print(file)
        #print(np.shape(split_data))


        """

        # denoise every single block and save it in buffer
        for i in range(np.shape(split_data)[0]):
            denoised_data = denoise_data_DASDL(split_data[i], model)
            np.save("experiments/15_DASDL/buffer/denoised_buffer" + str(i+1) + ".npy", denoised_data)
            gc.collect()


        # Save Data
        # Liste zum Speichern der geladenen Blöcke
        loaded_blocks = []

        # Laden der Blöcke in der richtigen Reihenfolge
        for i in range(9):  # Angenommen, es sind 9 Blöcke
            block_path = f"experiments/15_DASDL/buffer/denoised_buffer{i + 1}.npy"
            loaded_block = np.load(block_path)
            loaded_blocks.append(loaded_block)

        # Wieder zusammengeführter Array
        reconstructed_data = reassemble_blocks(loaded_blocks)
        print("reconstructed_data.shape = ", reconstructed_data.shape)
        write_das_h5.write_block(reconstructed_data, headers, denoised_file_path)
        
        """

    else:
        print("File " + str(i_file), ": " + file, " already denoised. No Denoising is performed.")



    gc.collect()

