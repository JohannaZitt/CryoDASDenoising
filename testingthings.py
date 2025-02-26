

'''

Script 1: Generating .txt file for batch script (Splited Data)


'''
import os

import matplotlib.pyplot as plt
import numpy as np

""" 
import os

data_folder_path = "/media/johanna/Elements/DLDAS_Denoising/split_numpy_das_data"
data_files = os.listdir(data_folder_path)

with open("experiments/15_DASDL/terminal_commands.txt", "a") as file:

    for data_file in data_files:
        print_text = "matlab -nodisplay -batch \"A_computing_cwt('/media/johanna/Elements/DLDAS_Denoising/split_numpy_das_data/" + data_file + "', '/media/johanna/Elements/DLDAS_Denoising/cwt_data/cwt_" + data_file[:-4] + ".mat')\""
        file.write(print_text + "\n")

"""

"""

Skript 1.2: Generating .txt file for batch scipt (cut to size)




import os

data_folder_path = "experiments/15_DASDL/test_data"
data_files = os.listdir(data_folder_path)

with open("experiments/15_DASDL/terminal_commands.txt", "a") as file:

    for data_file in data_files:
        print_text = "matlab -nodisplay -batch \"A_computing_cwt('/home/johanna/PycharmProjects/MAIN_DAS_denoising/experiments/15_DASDL/test_data/" + data_file + "', '/home/johanna/PycharmProjects/MAIN_DAS_denoising/experiments/15_DASDL/cwt_scale_data/cwt_" + data_file[:-4] + ".mat')\""
        file.write(print_text + "\n")

"""


'''

Script 2: Testing reading data from .mat format


'''

"""

import scipy.io as sio

# Load the .mat file
data = sio.loadmat('/media/johanna/Elements/DLDAS_Denoising/cwt_data/cwt_rhone1khz_UTC_20200727_001708.575.h5_0.mat')

# Print the keys to see what variables are in the file
print(data.keys())

# Access your variables
data_bp = data['outF']  # Replace 'outF' with actual variable name
data_cwt = data['out']  # Replace 'out' with actual variable name
data = data['dn']

print(data_bp.shape)

"""


'''

Skript 3: rearange denoised blocks to one single .h5 file

'''


import numpy as np
import matplotlib.pyplot as plt
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5
from pydas_readers.readers.load_das_h5_CLASSIC import load_headers_only
from helper_functions import resample


def reassemble_blocks(blocks):
    side = int(np.sqrt(len(blocks)))
    rows = []
    for i in range(side):
        row_blocks = blocks[i*side:(i+1)*side]
        row = np.concatenate(row_blocks, axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def merge_blocks(blocks, n=9):
    """
    Merge blocks back together that were previously split using split_array function.

    Parameters:
    blocks (list): List of numpy arrays (blocks) to merge
    n (int): Number of blocks (default=9, resulting in 3x3 grid)

    Returns:
    numpy.ndarray: Merged array with original dimensions
    """
    if len(blocks) != n:
        raise ValueError(f"Expected {n} blocks, but got {len(blocks)}")

    side = int(np.sqrt(n))

    # Get dimensions of final array
    block_height = blocks[0].shape[0]
    block_width = blocks[0].shape[1]
    total_height = block_height * side
    total_width = block_width * side

    # Create empty array to hold merged result
    merged = np.empty((total_height, total_width), dtype=blocks[0].dtype)

    # Place each block in the correct position
    for i in range(n):
        row = i // side
        col = i % side
        h_start = row * block_height
        h_end = (row + 1) * block_height
        w_start = col * block_width
        w_end = (col + 1) * block_width

        merged[h_start:h_end, w_start:w_end] = blocks[i]

    return merged

def reshape_das(data, headers):

    #print(data.shape)

    npts = data.shape[0]
    nchan = data.shape[1]

    res = data

    # downsample in space:
    if nchan == 4864 or nchan == 4800 or nchan == 4928:
        res = data[:, ::6]
    elif nchan == 2496:
        res = data[:, ::3]
    else:
        print("Unkown nchan: ", res.shape[1])


    # downsample in time:
    res = resample(res, headers['fs'] / 400)

    # update header:
    headers["npts"] = res.shape[0]
    headers["nchan"] = res.shape[1]
    headers["fs"] = 400
    headers["dx"] = 12


    #print("RESHAPE_DATA.SHAPE: ", res.shape)

    return res, headers




folder_path = "experiments/15_DASDL/denoisedDAS_image/pieces/"
files = os.listdir(folder_path)
files_0 = [file for file in files if file.endswith("_0.npy")]

raw_file_names = ["rhone1khz_UTC_20200727_002138.575.h5", "rhone1khz_UTC_20200727_194308.575.h5",  "rhone1khz_UTC_20200727_050438.575.h5"]

#print(files_0)

for j, file in enumerate(files_0):

    print("\n\n")

    denoised_file_path = "experiments/15_DASDL/denoisedDAS_image/complete/"
    das_file_name = file[19:55]

    if not os.path.exists(denoised_file_path + "denoised_DASDL_" + das_file_name):

        print(file)

        loaded_blocks = []

        # Load Blocks in correct order
        for i in range(9):
            block_path = folder_path + file[:-9] + ".h5_" + str(i) + ".npy"
            loaded_block = np.load(block_path)
            loaded_block = loaded_block.T
            loaded_blocks.append(loaded_block)

        reconstructed_data = merge_blocks(loaded_blocks)

        # Save as .npy file
        # np.save(denoised_file_path + das_file_name + ".npy", reconstructed_data)
        print(raw_file_names[j])
        headers = load_headers_only("data/raw_DAS_image/" + raw_file_names[j])
        print(headers)

        das_data, das_headers = reshape_das(reconstructed_data.T, headers)

        print(das_headers)
        #print(das_data.shape)

        print("restoring file: ", file, ";  with shape: ", das_data.shape)
        write_das_h5.write_block(das_data, das_headers, denoised_file_path + "denoised_DASDL_" + das_file_name + ".h5")

    else:
        print("File ", file, " already exists. ")



'''  

Skript 4: plotting the denoised data as a hole



import matplotlib.pyplot as plt
import numpy as np

data = np.load("experiments/15_DASDL/denoisedDAS/rhone1khz_UTC_20200727_050438.575.h5.npy")
print(data.shape)

# Plot erstellen
plt.figure(figsize=(12, 8))  # Größe des Plots anpassen
plt.imshow(data, extent=(0 ,2500,0,3000), vmin=-1000, vmax=1000)
#plt.colorbar()  # Farbskala anzeigen

# Achsenbeschriftungen
plt.xlabel('Kanal')
plt.ylabel('Zeit')

# Plot anzeigen
plt.show()
'''

'''
Skript 5: plotting the raw data as a hole



import matplotlib.pyplot as plt
import numpy as np
import h5py

file = "data/raw_DAS/rhone1khz_UTC_20200727_050438.575.h5"
with h5py.File(file, "r") as f:
    data_raw  = f["Acoustic"][:]
print(data.shape)

# Plot erstellen
plt.figure(figsize=(12, 8))  # Größe des Plots anpassen
plt.imshow(data_raw, extent=(0 ,2500,0,3000), vmin=-10, vmax=10)
#plt.colorbar()  # Farbskala anzeigen

# Achsenbeschriftungen
plt.xlabel('Kanal')
plt.ylabel('Zeit')

# Plot anzeigen
plt.show()
'''


'''  

Skript 6: plotting the nine sections seperately



import matplotlib.pyplot as plt
import numpy as np

for i in range(9):

    print(i)

    data = np.load("/media/johanna/Elements/DLDAS_Denoising/denoised_data/denoised_DASDL_cwt_rhone1khz_UTC_20200727_002138.575.h5_" + str(i) + ".mat.npy")
    print(data.shape)

    # Plot erstellen
    plt.figure(figsize=(12, 8))  # Größe des Plots anpassen
    plt.imshow(data, extent=(0 ,2500,0,3000))
    #plt.colorbar()  # Farbskala anzeigen

    # Achsenbeschriftungen
    plt.xlabel('Kanal')
    plt.ylabel('Zeit')

    # Plot anzeigen
    plt.show()

'''

'''

Skript 7: plotting generated Data by Omar



file = "experiments/15_DASDL/denoisedDAS_omar/Denoised_Johanna_Matlab_CWT.npy"

data = np.load(file)
print(data.shape)

cmap = "plasma"
t_start_das = 0
t_end_das = data.shape[0]
ch_start = 0
ch_end = data.shape[1]
ch_spacing = 4

plt.imshow(data, cmap=cmap, extent=(0 ,(t_end_das-t_start_das)/1000,0,ch_end * ch_spacing/1000))
plt.show()


'''