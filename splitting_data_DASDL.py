import h5py
import numpy as np
import os
from tensorflow import keras

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5


def split_array(arr, n=9):
    side = int(np.sqrt(n))
    return [
        arr[i // side * arr.shape[0] // side:(i // side + 1) * arr.shape[0] // side,
        (i % side) * arr.shape[1] // side:(i % side + 1) * arr.shape[1] // side]
        for i in range(n)
    ]

def save_split_arrays(split_data, prefix="block"):
    for i, block in enumerate(split_data):
        filename = f"{prefix}_{i}.npy"
        np.save(filename, block)
        print(f"Saved: {filename}")


files_path = "data/raw_DAS_image"
files = os.listdir(files_path)

"""  Load model  """
model = keras.models.load_model("experiments/15_DASDL/DASDL_models/DAS_PATCH_Johanna.h5")
# model.summary()

""" Parameters """
lowcut = 0.001
highcut = 120
order = 4
print_count = 0

for i_file, file in enumerate(files):

    denoised_file_path = "experiments/15_DASDL/test/denoised_DASDL_" + file + ".h5"

    """  Load DAS data """
    file_path = os.path.join(files_path, file)
    with h5py.File(file_path, "r") as f:
        data = f["Acoustic"][:]
    headers = load_das_h5.load_headers_only(file_path)
    fs = headers["fs"]

    # ensure channel spacing of 4 m
    n_ch_orig = data.shape[0]
    if n_ch_orig == 4800 or n_ch_orig == 4864 or n_ch_orig == 4928:
        data = data[::2, :]

    # split data for denoising
    split_data = split_array(data)

    # save blox
    save_split_arrays(split_data, prefix="experiments/15_DASDL/cwt_data/numpy_data/" + file)



