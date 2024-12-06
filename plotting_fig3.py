import os
import re

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import compute_moving_coherence



def extract_id(filename):
    id = re.search(r"ID:(\d+)", filename).group(1)
    return id


def extract_SNR(filename):
    snr_pattern = re.compile(r"SNR:(\d+(\.\d+)?)\.npy")
    match = re.search(snr_pattern, filename)
    return float(match.group(1))



"""

Here Figure 3 is generated


"""


""" Parameters """

cmap = "plasma"
vmin = -1.5
vmax = 1.5
ch_start = 10
ch_end = 70
ch_total = 60


t_start_wiggle = 1050
t_end_wiggle = 1250
channel_wiggle_comparison = 28

event_id = 46
SNR_values = [0.0, 1.0, 3.2, 10.0]

col_pink = "#CE4A75"
fs=16 # fontsize
delta = 2 # fontsize delta

""" Get Raw Data Names"""
data_path = "data/synthetic_DAS/from_seis"
event_names_all = os.listdir(data_path)
event_names = [event_name for event_name in event_names_all if str(event_id) == extract_id(event_name)]
remove_events = []
for event_name in event_names:
    if not extract_SNR(event_name) in SNR_values:
        remove_events.append(event_name)
for event in remove_events:
    event_names.remove(event)
event_names.sort(key=extract_SNR)
first_event = event_names.pop(0)
event_names.append(first_event)
event_names = event_names[::-1]

""" Get Denoised Data Names"""
experiment = "03_accumulation_horizontal"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)

""" Load Ground Truth Data"""
ground_truth_data = np.load(os.path.join(data_path, "clean_ID:46_SNR:0.npy"))[ch_start:ch_end]


""" Create Plot """
fig, axs = plt.subplots(len(event_names), 4, figsize=(12, 14), gridspec_kw={"width_ratios": [5, 5, 1, 5]})

for i, event_name in enumerate(event_names):

    """ Load Data """
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[ch_start:ch_end]

    """ Calculate CC """
    bin_size = 11
    raw_cc = compute_moving_coherence(data, bin_size)
    denoised_cc = compute_moving_coherence(denoised_data, bin_size)
    raw_denoised_cc = denoised_cc / raw_cc
    raw_denoised_cc = raw_denoised_cc[ch_start:ch_end]
    raw_denoised_cc = raw_denoised_cc[::-1]
    x = np.arange(raw_denoised_cc.shape[0])
    y_seis = raw_denoised_cc
    X_seis = np.vstack((x, y_seis)).T
    X_seis = np.vstack((X_seis[:, 1], X_seis[:, 0])).T

    """ Plotting Data: """
    axs[i, 0].imshow(data, cmap=cmap, aspect="auto", interpolation="antialiased",
          vmin=vmin, vmax=vmax)
    axs[i, 1].imshow(denoised_data, cmap=cmap, aspect="auto", interpolation="antialiased",
                     vmin=vmin, vmax=vmax)
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color = "black")
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")

    """ Plotting wiggle for wiggle comparison """
    axs[i, 3].plot(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color=col_pink,
                       label="Co-Located Seismometer", linewidth=1.5, alpha=0.8, zorder=1)
    if not i == 0:
        axs[i, 3].plot(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey", label="Synthetics",
                       linewidth=1.5, alpha=0.6, zorder=1)
    axs[i, 3].plot(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
                   label="Denoised", linewidth=1.5, alpha=0.8, zorder=1)

    # print max. amplitudes:
    # print("Event " + event_name)
    # print("maximal amplitude of ground truth data: ", str(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle].max()))
    # print("maximal amplitude of noisy data: ", str(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle].max()))
    # print("maximal amplitude of denoised data: ", str(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle].max()))

    # legend
    #axs[i, 3].legend(fontsize = 15)


    """ Label and Ticks """
    axs[i, 0].set_ylabel("Offset [km]", fontsize=fs)
    axs[i, 0].set_yticks([59, 49 ,39, 29, 19, 9, 0], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize =fs-delta)
    axs[i, 1].set_yticks([59, 49 ,39, 29, 19, 9, 0], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=fs-delta)
    axs[i, 1].set_yticklabels([])
    axs[i, 2].set_yticks([])
    axs[i, 2].set_xlim(0, 9)
    axs[i, 2].set_ylim(0, raw_denoised_cc.shape[0]-1)
    axs[i, 3].set_yticks([])

    axs[i, 0].set_xticks([400, 800, 1200, 1600, 2000], [1, 2, 3, 4, 5], fontsize = fs-delta)
    axs[i, 1].set_xticks([400, 800, 1200, 1600, 2000], [1, 2, 3, 4, 5], fontsize = fs-delta)
    axs[i, 2].set_xticks([1, 7], [1, 7], fontsize = fs-delta)
    axs[i, 3].set_xticks([50, 100, 150], [0.2, 0.3, 0.4], fontsize = fs-delta)
    if i == 3:
        axs[i, 0].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 1].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 2].set_xlabel("Gain []", fontsize=fs)
        axs[i, 3].set_xlabel("Time [s]", fontsize=fs)
    else:
        axs[i, 0].set_xticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 2].set_xticklabels([])
        axs[i, 3].set_xticklabels([])

    ax2 = axs[i, 3].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("Amplitude [norm.]", fontsize=fs)

    """ plot arrows """
    arrow_style = "fancy,head_width=0.5,head_length=1.8"
    axs[i, 0].annotate("", xy=(0, ch_total - channel_wiggle_comparison),
                       xytext=(-0.05, ch_total - channel_wiggle_comparison),
                       arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))


axs[3, 0].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 0].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))


axs[0, 0].set_title("Synthetics", fontsize=fs+4, y=1.05)
axs[0, 1].set_title("Denoised", fontsize=fs+4, y=1.05)
axs[0, 2].set_title("LWC", fontsize=fs+4, y=1.05)
axs[0, 3].set_title("Wiggle Comparison", fontsize=fs+4, y=1.05)


""" Add letters in plots: """
letter_params = {
       "fontsize": fs+2,
       "verticalalignment": "top",
       "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
   }
letters = ["a", "b", "c", "d", "e", "f ", "g", "h", "i ", "j ", "k", "l ", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]

for i in range(4):
   axs[i, 0].text(x=0.12, y=1, transform=axs[i, 0].transAxes, s=snr_values[i], **letter_params)
   for j in range(4):
       axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*4 + j], **letter_params)

plt.tight_layout()

""" Save plot """
#plt.savefig("plots/fig3.pdf", dpi=400)
plt.show()









