import gc
from sys import prefix

import h5py
import numpy as np
import scipy.io as sio
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


def denoise_data_DASDL(data_file, model):

    w1 = 48
    w2 = 48
    s1z = 8
    s2z = 8

    data = sio.loadmat(data_file)

    # Print the keys to see what variables are in the file
    # print(data.keys())

    # Access your variables
    data_bp = data['outF']
    data_cwt = data['out']
    data = data['dn']
    print("BP_shape: ", np.shape(data_bp)) # (channel, time) :((((( DAS HIER IST FALSCH GLAUBE ICH!!!

    # We did not normalize the data:
    # data_cwt_norm = data_cwt / np.max(np.abs(data_cwt))

    """ Patching the CWT SCALE """
    cwt_scale_patch = patch(data_cwt, w1, w2, s1z, s2z)
    cwt_scale_input = np.reshape(cwt_scale_patch, (cwt_scale_patch.shape[0], w1 * w2, 1))

    """ Patching the Band-pass Filtered Data """
    band_pass_patch = patch(data_bp, w1, w2, s1z, s2z)
    band_pass_input = np.reshape(band_pass_patch, (band_pass_patch.shape[0], w1 * w2, 1))

    """ Denoising Data """
    #start = time.time()
    out = model.predict([band_pass_input, cwt_scale_input], batch_size=32)  # Time and Frequency domain
    #end = time.time()

    """ Reshape """
    out = np.reshape(out, (out.shape[0], w1 * w2))
    outA = np.transpose(out)
    n1, n2 = np.shape(data_bp)
    outB = patch_inv(outA, n1, n2, w1, w2, s1z, s2z)
    denoised_data = np.array(outB)

    #print("Denoised Data Shape: ", denoised_data.shape)

    #denoised_data = denoised_data.T

    return  denoised_data


"""  Load model  """
model = keras.models.load_model("experiments/15_DASDL/DASDL_models/DAS_PATCH_Johanna.h5")
#model.summary()

""" Parameters """
lowcut = 0.001
highcut = 120
order = 4
print_count = 0

path_to_files = "experiments/15_DASDL/cwt_scale_data/cwt_ablation"
files = os.listdir(path_to_files)

for file in files:

    if not os.path.exists("experiments/15_DASDL/denoisedDAS/denoised_DASDL_" + file +  ".npy"):
        data_file = os.path.join(path_to_files, file)
        print("Denoising File: ", file)
        denoised_data = denoise_data_DASDL(data_file, model)
        np.save("experiments/15_DASDL/denoisedDAS/denoised_DASDL_" + file +  ".npy", denoised_data)
        gc.collect()
    else:
        print("File denoised_DASDL_" + file +  ".npy already exists.")


    """
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