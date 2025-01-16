

'''

Script 1: Generating .txt file for batch script


'''
import os

"""
import os

data_folder_path = "/media/johanna/Elements/DLDAS_Denoising/split_numpy_das_data"
data_files = os.listdir(data_folder_path)

with open("experiments/15_DASDL/terminal_commands.txt", "a") as file:

    for data_file in data_files:
        print_text = "matlab -nodisplay -batch \"A_computing_cwt('/media/johanna/Elements/DLDAS_Denoising/split_numpy_das_data/" + data_file + "', '/media/johanna/Elements/DLDAS_Denoising/cwt_data/cwt_" + data_file[:-4] + ".mat')\""
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




import numpy as np
import matplotlib.pyplot as plt
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5
from pydas_readers.readers.load_das_h5_CLASSIC import load_headers_only

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

folder_path = "/media/johanna/Elements/DLDAS_Denoising/denoised_data/"
files = os.listdir(folder_path)
files_0 = [file for file in files if file.endswith("_0.mat.npy")]

print(files_0)

for file in files_0:

    loaded_blocks = []

    # Load Blocks in correct order
    for i in range(9):
        block_path = folder_path + file[:-9] + str(i) + ".mat.npy"
        #print(block_path)
        loaded_block = np.load(block_path)
        loaded_block = loaded_block.T
        loaded_blocks.append(loaded_block)

    reconstructed_data = merge_blocks(loaded_blocks)
    denoised_file_path = "experiments/15_DASDL/denoisedDAS/"
    das_file_name = file[19:55]

    # Save as .npy file
    np.save(denoised_file_path + das_file_name + ".npy", reconstructed_data)

    headers = load_headers_only("data/raw_DAS/" + das_file_name)

    print(headers)
    print(reconstructed_data.shape)
    print(type(reconstructed_data))

    write_das_h5.write_block(reconstructed_data, headers, denoised_file_path + "denoised_DASDL_" + das_file_name)

'''


'''  

Skript 4: plotting the denoised data as a hole

'''

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

Skript 4: plotting the nine sections seperately



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







