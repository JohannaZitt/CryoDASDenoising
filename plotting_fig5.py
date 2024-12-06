
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import load_das_data


"""

Here Figure 5 is generated


"""

""" Event IDs: """
# ID : [starttime, channel middle, channel delta, category, closts seismometer]
# Category 1: improved SNR
# Category 2: new event detected
# Category 3: worsening of an icequake through denoising
# Category 4: no event
event_times = {0: ["2020-07-27 08:17:34.6", 60, 70, 1, "ALH"],
               1: ["2020-07-27 12:07:48.3", 60, 40, 1, "ALH"],
               2: ["2020-07-27 14:14:17.3", 30, 30, 1, "ALH"],
               3: ["2020-07-27 04:58:26.0", 65, 60, 1, "ALH"],
               4: ["2020-07-27 05:08:55.0", 65, 60, 1, "ALH"],
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"],
               6: ["2020-07-27 04:05:23.0", 55, 30, 1, "ALH"],
               8: ["2020-07-27 14:52:07.0", 60, 25, 1, "ALH"],
               10: ["2020-07-27 02:10:09.4", 65, 70, 1, "ALH"],
               11: ["2020-07-27 19:43:01.4", 65, 65, 1, "ALH"],
               13: ["2020-07-27 02:29:19.8", 60, 40, 1, "ALH"],
               14: ["2020-07-27 04:05:49.6", 60, 35, 1, "ALH"],
               15: ["2020-07-27 13:43:53.7", 35, 15, 1, "ALH"],
               22: ["2020-07-27 07:07:07.7", 60, 25, 1, "ALH"],
               28: ["2020-07-27 03:04:41.0", 40, 15, 1, "ALH"],
               32: ["2020-07-27 04:02:04.0", 30, 10, 1, "ALH"],
               34: ["2020-07-27 03:46:03.5", 60, 70, 1, "ALH"],
               35: ["2020-07-27 03:03:20.2", 65, 55, 1, "ALH"],
               39: ["2020-07-27 06:28:41.3", 65, 40, 1, "ALH"],
               41: ["2020-07-27 04:05:34.0", 40, 10, 1, "ALH"],
               48: ["2020-07-27 01:31:03.0", 35, 20, 1, "ALH"],
               62: ["2020-07-27 06:30:28.0", 70, 35, 1, "ALH"],
               66: ["2020-07-27 08:30:59.4", 40, 5, 1, "ALH"],
               74: ["2020-07-27 04:47:30.8", 30, 10, 1, "ALH"],
               83: ["2020-07-27 01:03:00.2", 70, 25, 1, "ALH"],
               101: ["2020-07-27 00:41:05.5", 40, 10, 1, "ALH"],
               104: ["2020-07-27 05:36:58.0", 50, 20, 1, "ALH"],
               108: ["2020-07-27 03:29:36.0", 50, 10, 1, "ALH"],
               119: ["2020-07-27 06:54:18.0", 70, 10, 1, "ALH"],

               7: ["2020-07-27 08:32:45.0", 60, 10, 2, "Do"],
               9: ["2020-07-27 16:39:55.0", 30, 10, 2, "Do"],
               16: ["2020-07-27 14:14:35.0", 20, 10, 2, "Do"],
               17: ["2020-07-27 14:14:59.0", 20, 10, 2, "Do"],
               18: ["2020-07-27 16:26:04.5", 80, 10, 2, "Do"],
               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"],
               21: ["2020-07-27 14:21:23.5", 60, 20, 2, "Do"],
               24: ["2020-07-27 05:21:48.4", 70, 90, 2, "ALH"],
               25: ["2020-07-27 07:18:53.0", 60, 20, 2, "Do"],
               36: ["2020-07-27 20:47:34.8", 40, 45, 2, "ALH"],
               40: ["2020-07-27 12:53:43.5", 60, 20, 2, "Do"],
               42: ["2020-07-27 14:14:27.0", 60, 10, 2, "Do"],
               45: ["2020-07-27 22:02:32.5", 60, 20, 2, "Do"],
               46: ["2020-07-27 09:31:36.0", 60, 10, 2, "Do"],
               50: ["2020-07-27 01:21:39.0", 30, 10, 2, "Do"],
               52: ["2020-07-27 20:00:30.4", 100, 85, 2, "ALH"],
               56: ["2020-07-27 02:46:33.0", 60, 20, 2, "Do"],
               59: ["2020-07-27 09:34:09.5", 60, 10, 2, "Do"],
               64: ["2020-07-27 19:07:29.0", 40, 20, 2, "Do"],
               65: ["2020-07-27 17:52:40.5", 40, 20, 2, "Do"],
               67: ["2020-07-27 23:17:54.0", 70, 35, 2, "ALH"],

               75: ["2020-07-27 03:01:38.0", 60, 20, 2, "Do"],
               78: ["2020-07-27 05:18:21.0", 60, 10, 2, "Do"],
               80: ["2020-07-27 23:39:21.0", 5, 20, 2, "Do"],
               85: ["2020-07-27 04:25:22.0", 60, 20, 2, "Do"],
               87: ["2020-07-27 09:59:10.0", 60, 20, 2, "Do"],
               90: ["2020-07-27 14:14:14.0", 60, 20, 2, "Do"],
               94: ["2020-07-27 14:16:17.0", 60, 10, 2, "Do"],
               95: ["2020-07-27 13:27:41.8", 60, 20, 2, "Do"],
               96: ["2020-07-27 00:18:50.0", 60, 5, 2, "Do"],
               98: ["2020-07-27 23:23:39.0", 10, 20, 2, "Do"],
               102: ["2020-07-27 15:38:51.0", 60, 10, 2, "Do"],
               105: ["2020-07-27 07:13:20.0", 60, 10, 2, "Do"],
               107: ["2020-07-27 01:25:19.6", 30, 15, 2, "ALH"],
               116: ["2020-07-27 08:05:19.0", 60, 20, 2, "Do"],
               118: ["2020-07-27 01:58:21.0", 60, 20, 2, "Do"],

               82: ["2020-07-27 05:04:54.8", 40, 140, 3, "ALH"],
               }

ids = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 22, 28, 32, 34, 35, 39, 41, 48, 62, 66, 74, 83, 101, 104, 108, 119,
 7, 9, 16, 17, 18, 20, 21, 24, 25, 36, 40, 42, 45, 46, 50, 52, 56, 59, 64, 65, 67, 75, 78, 80, 85, 87, 90, 94, 95, 96, 98, 102, 105, 107, 116, 118,
 82]

""" Experiment """
experiment = "03_accumulation_horizontal"

raw_path = os.path.join("data", "raw_DAS/")
denoised_path = os.path.join("experiments", experiment, "denoisedDAS/")

""" Create Plot"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fs_s=14
fs_m=16
fs_l=18

sum_raw_f_frq = np.zeros((401,))
sum_raw_f_amp = np.zeros((401,))
sum_denoised_f_frq = np.zeros((401,))
sum_denoised_f_amp = np.zeros((401,))

number = 0

for id in ids:
    number += 1

    event_time = event_times[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_end = t_start + timedelta(seconds=2)
    middle_channel = event_times[id][1]
    fs = 400

    """ load DAS data """
    raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end,
                                                    raw=True,
                                                    channel_delta_start=event_times[id][1],
                                                    channel_delta_end=event_times[id][2])
    denoised_data, denoised1_headers, denoised1_axis = load_das_data(denoised_path, t_start, t_end,
                                                                     raw=False,
                                                                     channel_delta_start=event_times[id][1],
                                                                     channel_delta_end=event_times[id][2])



    """ Calculating frequency content of one channel """

    # rfft = real Fast Fourier Transform for one channel -> converting to spectral domain
    # abs = absolute values of the complex numbers of fft, which represents the amplitude of each frequency component
    # raw_f_amp = the amlplitudes spectrum of one channel
    # 1/fs computes time interval between successive samples
    # raw_f_frq holds the array of frequency bins corrsponging to the amplitudes in raw_f_amp
    raw_f_amp = np.abs(np.fft.rfft(raw_data[middle_channel])) #shape = (401,)
    raw_channel_size = raw_data[middle_channel].size # = 801 data points in time
    raw_f_frq = np.fft.rfftfreq(raw_channel_size, 1./fs) # shape = (401, )
    denoised_f_amp = np.abs(np.fft.rfft(denoised_data[middle_channel]))
    denoised_size = denoised_data[middle_channel].size
    denoised_f_frq = np.fft.rfftfreq(denoised_size, 1. / fs)


    """ Calculating frequency content for multiple channels """

    raw_data_freq = np.zeros((raw_data.shape[0], raw_f_amp.shape[0]))
    for i in range(raw_data.shape[0]):
        raw_data_freq[i] = np.abs(np.fft.rfft(raw_data[i]))
    raw_data_freq = raw_data_freq[:, 0:250]
    denoised_data_freq = np.zeros((denoised_data.shape[0], denoised_f_amp.shape[0]))
    for i in range(denoised_data.shape[0]):
        denoised_data_freq[i] = np.abs(np.fft.rfft(denoised_data[i]))
    denoised_data_freq = denoised_data_freq[:, 0:250]

    """ Plot Frequency Content """
    ax1.plot(raw_f_frq, raw_f_amp, "k-", linewidth=1, alpha=0.1)
    ax2.plot(denoised_f_frq, denoised_f_amp, "k-", linewidth=1, alpha=0.1)

    sum_raw_f_frq += raw_f_frq
    sum_raw_f_amp += raw_f_amp
    sum_denoised_f_frq += denoised_f_frq
    sum_denoised_f_amp += denoised_f_amp


mean_raw_f_frq = sum_raw_f_frq / len(ids)
mean_raw_f_amp = sum_raw_f_amp / len(ids)
mean_denoised_f_frq = sum_denoised_f_frq / len(ids)
mean_denoised_f_amp = sum_denoised_f_amp / len(ids)

""" Specifying Axes"""
ax1.plot(mean_raw_f_frq, mean_raw_f_amp, color="red")
ax1.set_xlabel("Frequency [Hz]", fontsize=fs_m)
ax1.set_ylabel("Amplitude [norm.]", fontsize=fs_m)
ax1.set_title("Noisy", fontsize=fs_l)
ax1.set_ylim(0, 120)
ax1.set_yticks([])
ax1.tick_params(axis='both', which='major', labelsize=fs_s)

ax2.plot(mean_denoised_f_frq, mean_denoised_f_amp, color="red")
ax2.set_yticks([])
ax2.set_xlabel("Frequency [Hz]", fontsize=fs_m)
ax2.set_title("Denoised", fontsize=fs_l)
ax2.set_ylim(0, 120)
ax2.tick_params(axis='both', which='major', labelsize=fs_s)

""" Add letters in plots: """
letter_params = {
    "fontsize": 20,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w"}
}
ax1.text(x=0.0, y=1.0, transform=ax1.transAxes, s="A", **letter_params)
ax2.text(x=0.0, y=1.0, transform=ax2.transAxes, s="B", **letter_params)

""" Save plot """
plt.tight_layout()
plt.show()
#plt.savefig("plots/fig5.pdf", dpi=300)








