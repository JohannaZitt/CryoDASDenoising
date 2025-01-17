import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy import read

from helper_functions import load_das_data, butter_bandpass_filter, compute_moving_coherence

"""

Here Figure 4 is generated


"""

def normalize_minmax(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1
def normalize_by_max(data):
    abs_max = np.max(np.abs(data))
    return data / abs_max

""" Event IDs"""
# specify event ID: [event time, start_channel, amount_channel, category, receiver]
event_times = {0: ["2020-07-27 08:17:34.5", 40, 40, 1, "ALH"],
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"],

               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"],

               82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH"],
               }

""" Experiment"""
# "01_ablation_horizontal" #
# "03_accumulation_horizontal"
# "07_retrained_combined200"
# "11_vanende"
# "12_vanende_finetuned_cryo"
# "13_isken_filter"
# "14_julius_filter"
# "15_DASDL"
experiment = "01_ablation_horizontal"

raw_path = os.path.join("data", "raw_DAS/")
denoised_path = os.path.join("experiments", experiment, "denoisedDAS/")

ids = [5, 20, 82]

""" Create Plot """
fig, axs = plt.subplots(len(ids), 4,
                       gridspec_kw={
                           "width_ratios": [5, 5, 1, 5],
                           "height_ratios": [1, 1, 1]},
                      sharey = False)
fig.set_figheight(10)
fig.set_figwidth(12)

for i, id in enumerate(ids):

    event_time = event_times[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_end = t_start + timedelta(seconds=2)

    """ Load Seismometer Data: """
    string_list = os.listdir("data/test_data/accumulation/")
    if id == 5:
        filtered_strings = [s for s in string_list if s.startswith("ID:5_")]
    else:
        filtered_strings = [s for s in string_list if s.startswith("ID:"+str(id))]

    seis_data_path = "data/test_data/accumulation/" + filtered_strings[0]
    seis_stream = read(seis_data_path, starttime=UTCDateTime(t_start),
                       endtime=UTCDateTime(t_end))
    seis_data = seis_stream[0].data
    seis_stats = seis_stream[0].stats
    seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
    #seis_data = seis_data / np.std(seis_data)
    seis_data = normalize_by_max(seis_data)


    """ Load DAS Data: """
    raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end, raw=True, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])
    denoised_data, denoised_headers, denoised_axis = load_das_data(denoised_path, t_start, t_end, raw=False, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])


    """ Normalize Data for Plotting Reasons: """
    raw_data_norm = normalize_by_max(raw_data)
    denoised_data_norm = normalize_by_max(denoised_data)

    print("\n\n")
    print("ID: ", id)
    print("RAW: ")
    print("Shape: ", raw_data.shape)
    print("Headers: ", raw_headers)
    print("DENOISED: ")
    print("Shape: ", denoised_data.shape)
    print("Headers: ", denoised_headers)



    """ Calculate CC """
    bin_size = 11
    raw_cc = compute_moving_coherence(raw_data_norm, bin_size)
    denoised_cc = compute_moving_coherence(denoised_data_norm, bin_size)
    raw_denoised_cc = denoised_cc / raw_cc
    raw_denoised_cc = raw_denoised_cc[::-1]

    """ Parameters for Plotting """
    cmap = "plasma" # verschiednene colormaps:  cividis, plasma, inferno, viridis, magma, (cmocean.cm.curl, seismic)
    t_start_das = 0
    t_end_das = denoised_data_norm.shape[1]
    ch_start = 0
    ch_end = denoised_data_norm.shape[0]
    channels = raw_data_norm.shape[0]
    middle_channel = event_times[id][1]
    ch_ch_spacing = 12
    vmin=-0.9
    vmax=0.9
    fs = 16

    """ Plotting Raw Data: """
    axs[i, 0].imshow(raw_data_norm, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end_das-t_start_das)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=vmin, vmax=vmax)
    axs[i, 0].set_ylabel("Offset [km]", fontsize=fs)
    axs[i, 0].tick_params(axis='y', labelsize=fs-2)

    """ Plotting Denoised Data: """
    im = axs[i, 1].imshow(denoised_data_norm, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end_das-t_start_das)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=vmin, vmax=vmax)
    axs[i, 1].set_yticklabels([])

    #cbar=fig.colorbar(im, ax=axs[i, 1])
    #cbar.set_label("Strain Rate [norm.]", fontsize=fs)
    #cbar.ax.tick_params(labelsize=fs-2, rotation=90)
    #cbar.set_ticks([-6, 0, 6])
    #cbar.set_ticklabels(['-1', '0', '1'])

    """ Plotting CC Gain"""
    x = np.arange(ch_end-ch_start)
    y_seis = raw_denoised_cc[ch_start:ch_end]
    X_seis = np.vstack((x, y_seis)).T
    X_seis = np.vstack((X_seis[:, 1], X_seis[:, 0])).T
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color="black")
    axs[i, 2].invert_yaxis()
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")
    axs[i, 2].set_ylim(0, raw_denoised_cc.shape[0]-1)
    axs[i, 2].set_xlim(0, 6)
    axs[i, 2].set_yticks([])
    axs[i, 2].set_yticklabels([])

    """ Plotting Wiggle comparison """
    col_pink = "#CE4A75"
    if id == 82:
        t_start_wiggle = 320
        t_end_wiggle = 480
    else:
        t_start_wiggle = 270
        t_end_wiggle = 430
    axs[i, 3].plot(raw_data_norm[middle_channel][t_start_wiggle:t_end_wiggle], color="grey", label="Noisy", linewidth=1.5, alpha=0.6, zorder=1)
    axs[i, 3].plot(denoised_data_norm[middle_channel][t_start_wiggle:t_end_wiggle], color="black", label="Denoised", linewidth=1.5, alpha=0.8, zorder=1)
    axs[i, 3].plot(seis_data[t_start_wiggle:t_end_wiggle], color=col_pink, label="Co-Located Seismometer", linewidth=1.5, alpha=0.8, zorder=1)
    axs[i, 3].set_yticks([])
    ax2 = axs[i, 3].twinx()

    ax2.set_ylabel("Amplitude [norm.]", fontsize=fs, color="black")
    ax2.set_yticks([])
    ax2.tick_params(axis="y", labelcolor="red")
    #axs[i, 3].legend(fontsize=15)

    """ Plot Arrow """
    arrow_style = "fancy,head_width=0.5,head_length=1.8"
    arrow_style2 = "fancy,head_width=0.5,head_length=0.5"
    axs[i, 0].annotate("", xy=(0, (channels - middle_channel) * 0.0125),
                       xytext=(-0.05, (channels - middle_channel) * 0.0125),
                       arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))

    # plot arrow in time domain:
    marker_position_1 = t_start_wiggle / 400
    marker_position_2 = t_end_wiggle / 400
    if i == 2:
        axs[i, 0].annotate("", xy=(marker_position_1, 0),
                           xytext=(marker_position_1, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=0.5))
        axs[i, 1].annotate("", xy=(marker_position_1, 0),
                           xytext=(marker_position_1, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=0.5))
        axs[i, 0].annotate("", xy=(marker_position_2, 0),
                           xytext=(marker_position_2, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=0.5))
        axs[i, 1].annotate("", xy=(marker_position_2, 0),
                           xytext=(marker_position_2, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=0.5))
    else:
        axs[i, 0].annotate("", xy=(marker_position_1, 0),
                           xytext=(marker_position_1, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style2, linewidth=0.5))
        axs[i, 1].annotate("", xy=(marker_position_1, 0),
                           xytext=(marker_position_1, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style2, linewidth=0.5))
        axs[i, 0].annotate("", xy=(marker_position_2, 0),
                           xytext=(marker_position_2, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style2, linewidth=0.5))
        axs[i, 1].annotate("", xy=(marker_position_2, 0),
                           xytext=(marker_position_2, -0.03),
                           arrowprops=dict(color="black", arrowstyle=arrow_style2, linewidth=0.5))

""" Set Titles """
axs[0, 0].set_title("Noisy", y=1.0, fontsize=fs+2)
axs[0, 1].set_title("Denoised", y=1.0, fontsize=fs+2)
axs[0, 2].set_title("LWC", y=1.0, fontsize=fs+2)
axs[0, 3].set_title("Wiggle Comparison", y=1.0, fontsize=fs+2)
""" Set Ax Labels """
axs[2, 0].set_xlabel("Time [s]", fontsize=fs)
axs[2, 0].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)
axs[2, 1].set_xlabel("Time [s]", fontsize=fs)
axs[2, 1].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)
axs[2, 2].set_xlabel("Gain []", fontsize=fs)

axs[0, 2].set_xticks([1, 3, 5])
axs[0, 2].set_xticklabels([])
axs[1, 2].set_xticks([1, 3, 5])
axs[1, 2].set_xticklabels([])
axs[2, 2].set_xticks([1, 3, 5], [1, 3, 5], fontsize=fs-2)

axs[0, 0].set_xticks([0.5, 1.0, 1.5])
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticks([0.5, 1.0, 1.5])
axs[0, 1].set_xticklabels([])
axs[1, 0].set_xticks([0.5, 1.0, 1.5])
axs[1, 0].set_xticklabels([])
axs[1, 1].set_xticks([0.5, 1.0, 1.5])
axs[1, 1].set_xticklabels([])

axs[0, 3].set_xticks([40, 80, 120])
axs[0, 3].set_xticklabels([])
axs[1, 3].set_xticks([40, 80, 120])
axs[1, 3].set_xticklabels([])
axs[2, 3].set_xticks([40, 80, 120])
axs[2, 3].set_xticklabels([0.1, 0.2, 0.3], fontsize=fs-2)
axs[2, 3].set_xlabel("Time [s]", fontsize=fs)


""" Add letters in plots """
letter_params = {
    "fontsize": fs + 2,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w"}
}
letters = ["a", "b", "c", "d", "e", "f ", "g", "h", "i ", "j ", "k", "l ", "m", "n", "o", "p"]

for i in range(3):
    for j in range(4):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i * 4 + j], **letter_params)

""" Save Plot """
plt.tight_layout()
plt.show()
#plt.savefig("plots/comparison/fig4test.pdf", dpi=400)
