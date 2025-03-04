import os
from obspy import read, UTCDateTime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from helper_functions import get_middel_channel, butter_bandpass_filter, resample
import numpy as np

def load_das_data(folder_path, t_start, t_end, receiver, raw, ch_middle, ch_delta):

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
    #ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middle-ch_delta:ch_middle+ch_delta]

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
    fs = 400
    channels = raw_data.shape[0]
    time_samples = raw_data.shape[1]
    time_in_seconds = time_samples / fs

    font_s = 12
    font_m = 14
    font_l = 16
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
    for i in range(5):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xticks([], [])
    # plt.xlabel("Time[s]", size=font_m)
    plt.ylabel("Offset [km]", size=font_m)
    plt.yticks(np.arange(0, n * 1.5, 12), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], size=font_s)
    #plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.title("Noisy DAS Data", loc="left")
    plt.annotate("", xy=(0, (channels-middle_channel) * 1.5), xytext=(-1, (channels-middle_channel) * 1.5),
                 arrowprops=dict(color="red", arrowstyle="->", linewidth=2))

    """ Add red arrow """
    arrow_style = "fancy,head_width=0.5,head_length=1"
    ax = plt.gca()
    ax.annotate("", xy=(0, 22.5),
                xytext=(-0.02, 22.5),
                arrowprops=dict(color="red", arrowstyle=arrow_style, linewidth=2))

    """ Plot denoised data """
    plt.subplot(222)
    i = 0
    for ch in range(channels):
        plt.plot(denoised_data[-ch][:] + 1.5 * i, "-k", alpha=alpha)
        i += 1
    for i in range(5):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)

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
    for i in range(5):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    plt.xticks(np.arange(0, time_samples, 200), np.arange(0, time_in_seconds, 0.5), size=font_s)
    plt.ylabel("Seismometer", size=font_l)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """ Plot Seismometer data denoised """
    plt.subplot(224)
    plt.plot(seis_data, color="black", alpha=0.4)
    for i in range(5):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    plt.xticks(np.arange(0, time_samples, 200), np.arange(0, time_in_seconds, 0.5), size=font_s)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """ Save Figure """
    plt.tight_layout()
    #plt.show()
    plt.savefig(saving_path + ".pdf", bbox_inches="tight", pad_inches=0.5, dpi=400)

# TODO: do it again with the model "02_horizontal_accumulation"

seismometer_data_path = "data/test_data/accumulation_seismometer/"
seismometer_events = os.listdir(seismometer_data_path)
for seismometer_event in seismometer_events:
    print(seismometer_event)
    # Get event information from data path:
    date = "2020-07-27"
    receiver = "ALH"
    id = seismometer_event[:-9]
    ch_delta = 25
    ch_middle = int(3842/6) - 10

    time = seismometer_event[-23:-15] + ".0"
    t_start = datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M:%S.%f")
    t_start = t_start - timedelta(seconds=1.5)
    t_end = t_start + timedelta(seconds=3)

    # Load seismometer data:
    seismometer_stream = read(seismometer_data_path + seismometer_event, starttime=UTCDateTime(t_start), endtime=UTCDateTime(t_end))
    seis_stats = seismometer_stream[0].stats
    seismometer_data = seismometer_stream[0]

    # Load raw DAS data:
    raw_folder_path = "data/raw_DAS/"
    raw_data, raw_headers, raw_axis = load_das_data(folder_path=raw_folder_path, t_start=t_start, t_end=t_end,
                                                    receiver=receiver, raw=True, ch_middle=ch_middle, ch_delta=ch_delta)

    # Load denoised DAS data:
    denoised_folder_path = "experiments/03_combined200/denoisedDAS/"
    denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path=denoised_folder_path, t_start=t_start,
                                                                   t_end=t_end, receiver=receiver, raw=False, ch_middle=ch_middle, ch_delta=ch_delta)

    # Plots data:
    saving_path = "plots/test_data/" + id
    plot_sectionplot(raw_data=raw_data.T, denoised_data=denoised_data.T, seis_data=seismometer_data, seis_stats=seis_stats,
                     saving_path=saving_path, middle_channel=ch_middle, id=id)



