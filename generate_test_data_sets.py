# for calculating LWC, CC, and SNR values, we just need section of 10 seconds long and 50 channels
# DAS data sections.

import os
import re
import numpy as np
from datetime import datetime, timedelta
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from helper_functions import butter_bandpass_filter, get_middel_channel

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

def load_das_data(folder_path, t_start, t_end, receiver, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12

    # 3. cut to size
    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-40:ch_middel+40]
    if raw:
        # 4. downsample in time
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis

data_type = "ablation"
raw_folder_path = "data/raw_DAS/"
seismometer_path = "data/test_data/" + data_type + "_seismometer"
saving_path = "data/test_data/" + data_type + "_DAS"

events = os.listdir(seismometer_path)

for event in events:


    #print("SEISMOMETER EVENT: ", event)

    """ Search for correct event """
    event_time = event[-23:-15]
    event_date = event[-34:-24]
    id = re.search(r"ID:(\d+)", event).group(1)
    receiver = ""
    zone = ""

    if data_type[:2] == "ab":
        receiver = event[-14:-10]
        zone = "ablation"
    elif data_type[:2] == "ac":
        receiver = event[-12:-9]
        zone = "accumulation"
    else:
        print("ERROR: No matching data type")


    """ Pick Time Window """
    t_start = datetime.strptime(event_date + " " + event_time + ".0", "%Y-%m-%d %H:%M:%S.%f")
    t_start = t_start - timedelta(seconds=5)
    t_end = t_start + timedelta(seconds=10)
    ch_start = 17
    ch_end = 23

    """ Load raw DAS data """
    das_data = load_das_data(folder_path =raw_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = True)

    das_array = np.array(das_data[0])
    if not das_array.shape[0] == 4001:
        print(event)
        print("Shape: ", das_array.shape)

    save_name = "DAS_" + event[:-6] + "_" + data_type + ".npy"
    print(save_name)
    #np.save("data/test_data/" + data_type + "_DAS/" + save_name, das_array)
    #print("Eventdate: ", event_date, ";  Eventtime: ", event_time, "; Receiver: ", receiver,
    #      "; Zone: ", zone, "; DAS data type: ", type(das_data), "; Shape: ", das_data.shape[0])
