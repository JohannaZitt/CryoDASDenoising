
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from obspy import read
from obspy import UTCDateTime
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5

from helper_functions import get_middel_channel, butter_bandpass_filter, resample




def load_das_data(folder_path, t_start, t_end, receiver, raw, ch_delta_start, ch_delta_end):

    """

    loads DAS data

    """

    """ 1. load data """
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    """ 2. downsample data in space """
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12


    """ 3. cut to size """
    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-ch_delta_start:ch_middel+ch_delta_end]

    """ 4. downsample in time """
    if raw:
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    """ 5. bandpasfilter and normalize """
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        #data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis

def plot_sectionplot(raw_data, denoised_data, seis_data, seis_stats, saving_path, middle_channel, id):

    """

    Plots data as waveform section plot

    """

    """ Parameters """
    font_s = 12
    font_m = 14
    font_l = 16
    channels = raw_data.shape[0]
    fs = 400
    alpha = 0.7
    alpha_dashed_line = 0.2

    """ Create Plot"""
    fig, ax = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={"height_ratios": [7, 1]}, dpi=400)

    plt.rcParams.update({"font.size": font_l})
    plt.tight_layout()

    """ Plot raw data """
    plt.subplot(221)
    n = 0
    for ch in range(channels):
        plt.plot(raw_data[-ch][:] + 1.5 * n, "-k", alpha=alpha)
        n += 1
    for i in range(3):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xticks([], [])
    # plt.xlabel("Time[s]", size=font_m)
    plt.ylabel("Offset [km]", size=font_m)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))

    if id == 5:
        plt.yticks(np.arange(0, 190, 25), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], size=font_s)
    if id == 20:
        plt.yticks(np.arange(0, n * 1.5, 12), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=font_s)
    if id == 82:
        plt.yticks(np.arange(0, 350, 61), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], size=font_s)


    plt.title("Noisy DAS Data", loc="left")
    plt.annotate("", xy=(0, (channels-middle_channel) * 1.5), xytext=(-1, (channels-middle_channel) * 1.5),
                 arrowprops=dict(color="red", arrowstyle="->", linewidth=2))

    """ Plot denoised data """
    plt.subplot(222)
    i = 0
    for ch in range(channels):
        plt.plot(denoised_data[-ch][:] + 1.5 * i, "-k", alpha=alpha)
        i += 1
    for i in range(3):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)

    if id == 5:
        plt.yticks(np.arange(0, 190, 25), [], size=font_s)
    if id == 20:
        plt.yticks(np.arange(0, n * 1.5, 12), [], size=font_s)
    if id == 82:
        plt.yticks(np.arange(0, 350, 61), [], size=font_s)

    plt.xticks([], [])
    plt.title("Denoised DAS Data", loc="left")
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    plt.annotate("", xy=(0, (channels - middle_channel) * 1.5), xytext=(-1, (channels - middle_channel) * 1.5),
                 arrowprops=dict(color="red", arrowstyle="->", linewidth=2))
    plt.subplots_adjust(wspace=0.05)

    """ Plot Seismometer data raw """
    seis_fs = seis_stats["sampling_rate"]
    plt.subplot(223)
    plt.plot(seis_data, color="black", alpha=0.4)
    for i in range(3):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 1001, 250), np.arange(0, 2.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 801, 200), np.arange(0, 2.1, 0.5), size=font_s)
    plt.ylabel("Seismometer", size=font_l)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """ Plot Seismometer data denoised """
    plt.subplot(224)
    plt.plot(seis_data, color="black", alpha=0.4)
    for i in range(3):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 1001, 250), np.arange(0, 2.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 801, 200), np.arange(0, 2.1, 0.5), size=font_s)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """ Save Figure """
    plt.tight_layout()
    plt.show()
    #plt.savefig(saving_path + ".png", bbox_inches="tight", pad_inches=0.5, dpi=400)


"""

Here Figure S3-S5 is generated


"""


# ID : [starttime, start channel delta, end channel delta, category, closts seismometer]
events = {5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH", "5_"],
         20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH", "20"],
         82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH", "82"]
         }

""" Parameters """
id = 5 #set to 5 for generating Figure S3, set to 20 for generating Figure S4, set to 82 for generating Figure S5
event_time = events[id][0]
t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
t_end = t_start + timedelta(seconds=2)
experiment = "03_accumulation_horizontal"
receiver = "ALH"

""" load seismometer data """
string_list = os.listdir("data/test_data/accumulation/")
filtered_strings = [s for s in string_list if s.startswith("ID:" + events[id][5])]
seis_data_path = "data/test_data/accumulation/" + filtered_strings[0]
seis_stream = read(seis_data_path, starttime=UTCDateTime(t_start-timedelta(seconds=1)), endtime=UTCDateTime(t_end))
seis_data = seis_stream[0].data
seis_stats = seis_stream[0].stats
seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
seis_data = seis_data/np.abs(seis_data).max()
seis_data = seis_data[400:]

""" load raw DAS data """
raw_folder_path = "data/raw_DAS/"
raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start=t_start, t_end=t_end, receiver=receiver, raw=True, ch_delta_start=events[id][1], ch_delta_end=events[id][2])

""" load denoised DAS data """
denoised_folder_path = "experiments/" + experiment + "/denoisedDAS/"
denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path=denoised_folder_path, t_start=t_start, t_end=t_end, receiver=receiver, raw=False, ch_delta_start=events[id][1], ch_delta_end=events[id][2])

saving_path = "plots/figS5"

""" Create Plot """
plot_sectionplot(raw_data=raw_data.T, denoised_data=denoised_data.T, seis_data=seis_data, seis_stats=seis_stats, saving_path=saving_path, middle_channel=events[id][1], id=id)











