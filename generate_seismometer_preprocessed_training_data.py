import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from obspy import read

from helper_functions import butter_bandpass_filter as bandpass_filter
from helper_functions import resample_seis as resample



"""

Here we generate the initial waveforms from seismometer data for model training as described in Section 3.2 Model Training. 
The waveforms are converted to strain rate, bandpass filtered, downsampled and normalized by std.

The initial waveforms for the models "Ablation Horizontal", "Ablation Vertical", "Accumulation Horizontal", "Accumulation Vertical",
and "Borehole Seismometer" are generated and saved for model training.  

"""


###################
### Parameters: ###
###################

# downsample frequency:
fs: int = 400

# convert params:
gauge_length: int = 10
s_wave_velocity: int = 1800
t: float = gauge_length / s_wave_velocity
rollouttotal: int = int(t * fs) + 1
rollout = int(rollouttotal / 2)

# directories
savedir = "data/training_data/preprocessed_seismometer/"
folders = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical", "09_borehole_seismometer"]

for folder in folders:

    #####################
    ### Reading Data: ###
    #####################

    stream = read("data/training_data/raw_seismometer_trainingdata/" + folder + "/ID*.mseed")
    n_trc = len(stream)
    n_t = stream[0].stats.npts

    ########################
    ### Processing Data: ###
    ########################

    # 1. convert to np array
    data: ndarray = np.zeros((n_trc, n_t))
    for i in range(n_trc):
        data[i] = stream[i].data[0:n_t]


    for i in range(n_trc):
        # 2. convert to strain rate
        data[i] = np.roll(data[i], rollout) - np.roll(data[i], -rollout)
        data[i] /= gauge_length

        # 3. filter data
        data[i] = bandpass_filter(data[i], lowcut=1, highcut=120, fs=fs, order=4)

    # 4. downsample data to 400 Hz:
    fs_old = stream[0].stats.sampling_rate
    if fs_old != fs:
        data = resample(data, fs_old / fs)

    # 5. scale by std
    for i in range(n_trc):
        data[i] /= np.std(data[i])

    ######################
    ### Plotting Data: ###
    ######################

    #for i in range(10, 20):
    #    plt.plot(data[i])
    #    plt.show()

    #################################
    ### Saving Initial Waveforms: ###
    #################################
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    np.save(savedir + folder, data)


##################################################
### Compute Training Data for Combined Models: ###
##################################################

# Compute initial waveforms for combined200 model
arrays200 = []
for i in range(4):
    array200 = np.load("data/training_data/preprocessed_seismometer/" + folders[i] + ".npy")
    array200 = array200[0:50]
    arrays200.append(array200)
combined_array200 = np.vstack(arrays200)
np.save(savedir + "05_combined200", combined_array200)

# Compute initial waveform for combined800 model
arrays800 = []
for i in range(4):
    array800 = np.load("data/training_data/preprocessed_seismometer/" + folders[i] + ".npy")
    arrays800.append(array800)
combined_array = np.vstack(arrays800)
np.save(savedir + "06_combined800", combined_array)



