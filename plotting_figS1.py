import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from obspy import read

from helper_functions import butter_bandpass_filter


def resample(stream, ratio):
    """
    :param stream: stream object, which has to be resampled
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    # convert stream to np array
    n_trc = len(stream)
    n_t = stream[0].stats.npts
    data: ndarray = np.zeros((n_trc, n_t))
    for i in range(n_trc):
        data[i] = stream[i].data[0:n_t]

    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res



""" 

Here Figure S1 is generated

"""


""" Data Path """
event_path= "data/training_data/raw_seismometer_trainingdata/"
events=["01_ablation_horizontal/ID:0_2020-07-25_07:20:54_RA83_EH3.mseed",
        "01_ablation_horizontal/ID:49_2020-07-24_08:00:44_RA82_EH2.mseed",
        "02_ablation_vertical/ID:0_2020-07-23_13:10:53_RA81_EHZ.mseed",
        "02_ablation_vertical/ID:49_2020-07-26_15:59:44_RA84_EHZ.mseed",
        "03_accumulation_horizontal/ID:0_2020-07-23_11:45:30_c0AJP_p1.mseed",
        "03_accumulation_horizontal/ID:49_2020-07-23_19:32:29_c0AJP_p1.mseed",
        "04_accumulation_vertical/ID:0_2020-07-25_08:58:04_c0ALH_p0.mseed",
        "04_accumulation_vertical/ID:49_2020-07-26_19:41:17_c0ALH_p0.mseed",
        "09_borehole_seismometer/ID:0_2020-07-23_15:31:32_RA91_EH2.mseed",
        "09_borehole_seismometer/ID:49_2020-07-24_03:33:29_RA92_EHZ.mseed"]

""" Plotting Parameters """
title1 = "Highest Amplitude"
title2 = "Lowest Amplitude"
t_start = [1100, 1000, 850, 800, 950, 1100, 1100, 1100, 900, 1100]
t = 500
fontsize = 13
gain = 2
rotation = 90
letters = ["a", "b", "c", "d", "e", "f", "g",
           "h", "j", "k", "l", "m", "n", "o", "p"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

""" Create Plot """
fig, axes = plt.subplots(5, 2, figsize=(10,8))

for i, event in enumerate(events):

    """ Load Data """
    stream = read(os.path.join(event_path, event))

    """ Preprocessing Data """
    if not stream[0].stats["sampling_rate"] == 400:
        stream = resample(stream, stream[0].stats["sampling_rate"]/400)
    data = stream[0].data
    data = butter_bandpass_filter(data, lowcut=1, highcut=120, fs=400, order=4)
    data = data / np.std(data) # Data is normalized -> the amplitudes in the plot are distorted

    row = i // 2
    col = i % 2

    """ Plot Data """
    axes[row, col].plot(data[t_start[i]:t_start[i]+t], color="black", linewidth=1, alpha=0.8)
    axes[row, col].set_yticks([])
    axes[row, col].text(x=0.0, y=1.0, transform=axes[row, col].transAxes, s=letters[i], **letter_params)
    axes[row, col].set_ylim(-32, 32)

    """ Create Axes """
    axes[0, 0].set_ylabel("Ablation\nHorizontal",  fontsize=fontsize)
    axes[1, 0].set_ylabel("Ablation\nVertical",  fontsize=fontsize)
    axes[2, 0].set_ylabel("Accumulation\nHorizontal",  fontsize=fontsize)
    axes[3, 0].set_ylabel("Accumulation\nVertical",  fontsize=fontsize)
    axes[4, 0].set_ylabel("Borehole",  fontsize=fontsize)
    if row==0:
        axes[row, 0].set_title(title1, fontsize=fontsize+gain)
        axes[row, 1].set_title(title2, fontsize=fontsize+gain)

    if row==4:
        axes[row, col].set_xticks([0, 100, 200, 300, 400, 500], [0, 0.25, 0.5, 0.75, 1, 1.25], fontsize=fontsize)
        axes[row, col].set_xlabel("Time [s]", fontsize=fontsize)
    else:
        axes[row, col].set_xticks([])

""" Save Plot """
plt.tight_layout()
plt.show()
#plt.savefig("plots/figS1.pdf", dpi=250)
