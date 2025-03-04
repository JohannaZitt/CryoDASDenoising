import os
from datetime import datetime, timedelta

import numpy as np
from obspy import read

from helper_functions import butter_bandpass_filter
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5


def resample(data, ratio):
    try:
        res = np.zeros((int(data.shape[0]/ratio) + 1, data.shape[1]))
        for i in range(data.shape[1]):

            res[:,i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:,i])
    except ValueError as e:
        res = np.zeros((int(data.shape[0] / ratio), data.shape[1]))
        for i in range(data.shape[1]):
            res[:, i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:, i])
    return res

def load_das_data(folder_path, t_start, t_end, raw):

    """ Load Data """
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    """ Downsample Data in Space """
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::4]
        else:
            data = data[:, ::2]
        headers["dx"] = 8
    ch_middel = int(3460/4) # get start and end channel:
    data = data[:, ch_middel-40:ch_middel+40]

    """ Downsample Data in Time """
    if raw:
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    """ Filter and Normalize """
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        data[:,i] = data[:,i] / np.std(data[:,i])

    return data.T, headers, axis

def compute_shift(gauge = 12, slowness_type = "mean", fs = 400): # gauge is here channel spacing!
    slowness = 0
    if slowness_type == "max":
        slowness = 1 / 1650 # S-wave slowness
    if slowness_type == "mean":
        slowness = 1 / 2500
    if slowness_type == "min":
        slowness = 1 / 3900 # P-wave slowness

    shift = gauge * slowness * fs
    return round(shift)

"""

Here the data for the synthetic tests as described in Section 4.3 are generated


"""

fs = 400
lowcut = 1
highcut = 120
shift = compute_shift()
print(shift)
n_channel = 80
SNR_values = [-1, 0, 1, 2, 3, 4] #in dB


"""
GENERATE SYNTHETIC DATA FROM DAS DATA:
"""

das_folder_path = "data/synthetic_DAS/raw_DAS/"
event_times = ["20:18:58.0"]

ids = [34]

for n, event_time in enumerate(event_times):
    id = ids[n]
    for SNR in SNR_values:
        """ Load Data """
        t_start = datetime.strptime("2020-07-06 " + event_time, "%Y-%m-%d %H:%M:%S.%f")
        t_end = t_start + timedelta(seconds=6)
        das_data, headers, axis = load_das_data(das_folder_path, t_start, t_end, raw=True)

        """ Calculate Noise and Add Noise to Records """
        SNR = 10 ** (0.5 * SNR) #convert to linear SNR
        amplitude = 2 * SNR / np.abs(das_data).max()
        das_data = das_data * amplitude

        noise = np.random.standard_normal(das_data.shape)
        synthetic_data = das_data + noise

        for i in range(synthetic_data.shape[0]):
            """ Filter and Normalize Data """
            synthetic_data[i] = butter_bandpass_filter(synthetic_data[i], lowcut, highcut, fs, order=4)
            synthetic_data[i] = synthetic_data[i] / np.std(synthetic_data[i])

        """ Save Data """
        file_name = "DAS_ID:" + str(id) + "_SNR:" + str(round(SNR, 1))
        np.save("data/synthetic_DAS/from_DAS/" + file_name, synthetic_data)



"""
GENERATE SYNTHETIC DATA FROM SEISMOMETERS:
"""


folder_path = "data/synthetic_DAS/raw_seismometer/"
data_paths = os.listdir(folder_path)

for data_path in data_paths:

    """ Read Data """
    stream = read(folder_path + "/" + data_path)
    stats = stream[0].stats
    data = stream[0].data

    """ Filter and Normalize Data """
    data = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=4)
    data = data / np.std(data)

    """ Shift Data Accordingly to Moveout"""
    shifted_data = np.zeros((n_channel, data.shape[0]))
    for i in range(n_channel):
        shifted_data[i] = np.roll(data, i * shift)


    for SNR in SNR_values:
        """ Calculate Noise and add Noise """
        synthetic_data = shifted_data
        noise = np.random.standard_normal(synthetic_data.shape)

        SNR = 10 ** (0.5 * SNR)
        amplitude = 2 * SNR / np.abs(synthetic_data).max()
        for i in range(synthetic_data.shape[0]):
            synthetic_data[i] = synthetic_data[i] * amplitude

        synthetic_data = synthetic_data + noise

        for i in range(n_channel):
            """ Filter and Normalize Data """
            synthetic_data[i] = butter_bandpass_filter(synthetic_data[i], lowcut, highcut, fs, order=4)
            synthetic_data[i] = synthetic_data[i] / np.std(synthetic_data[i])

        """ Save Data """
        file_name = data_path.split("/")[-1][0:5] + "_SNR:" + str(round(SNR, 1)) + ".npy"
        np.save("data/synthetic_DAS/from_seis/" + file_name, synthetic_data)
        np.save("data/synthetic_DAS/from_seis/clean_ID:46_SNR:0.npy", shifted_data)

