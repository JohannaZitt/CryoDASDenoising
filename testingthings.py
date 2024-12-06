
import numpy as np


from datetime import datetime, timedelta
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from helper_functions import butter_bandpass_filter, compute_moving_coherence, xcorr, get_middel_channel



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

    print("data.shape1 ", data.shape)

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12
    print("data.shape2 ", data.shape)

    # 3. cut to size

    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-20:ch_middel+20]
    print("data.shape3 ", data.shape)
    if raw:
        # 4. downsample in time
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400
    print("data.shape4 ", data.shape)
    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        #data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        data[:, i] = data[:,i] / np.std(data[:, i])
    print("data.shape5 ", data.shape)
    return data, headers, axis




experiments = ["13_isken_filter"]

event_time = "22:56:54"

event_date = "2020-07-27"

t_start = datetime.strptime(event_date + " " + event_time + ".0", "%Y-%m-%d %H:%M:%S.%f")
t_start = t_start - timedelta(seconds=3)
t_end = t_start + timedelta(seconds=6)

receiver = "ALH"



for experiment in experiments:
    denoised_folder_path = "experiments/" + experiment + "/denoisedDAS/"
    denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path =denoised_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = False)
    print(denoised_data.shape)