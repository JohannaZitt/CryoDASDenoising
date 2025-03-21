import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy import read

from helper_functions import load_das_data, butter_bandpass_filter, compute_moving_coherence, resample


def normalize_minmax(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1
def normalize_by_max(data):
    abs_max = np.max(np.abs(data))
    return data / abs_max

"""

Here Figure 5 is generated: the denoised data for 

the denoised data for method 11_vanende_earth, 12_vanende_finetuned_cryo, 13_afk, 14_conventional, and 
15_DASDL have to be provided. The data is availabel at: 

"""


""" Event IDs"""
# ID: [event time, start_channel, amount_channel, category, receiver]
event_times = {0: ["2020-07-27 08:17:34.5", 40, 40, 1, "ALH"],
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"],

               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"],

               82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH"],
               }

""" Experiment"""
experiment_names = ["$\mathcal{J}$-invariant\ncryo", "$\mathcal{J}$-invariant\nearth+cryo", "$\mathcal{J}$-invariant\nearth", "DASDL", "AFK", "Conventional"]
experiments = ["02_accumulation", "12_vanende_finetuned_cryo", "11_vanende_earth", "15_DASDL", "13_afk", "14_conventional"] #
ids = [5, 20, 82]

for id in ids:

    """ Create Plot """
    fig, axs = plt.subplots(len(experiments), 3,
                           gridspec_kw={
                               "width_ratios": [5, 5, 6],
                               },
                          sharey = False)
    fig.set_figheight(18)
    fig.set_figwidth(13)
    fs = 20

    for i, experiment in enumerate(experiments):

        print("\n\n" + experiment)

        raw_path = os.path.join("data", "raw_DAS/")
        denoised_path = os.path.join("experiments", experiment, "denoisedDAS_image/")

        event_time = event_times[id][0]
        t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
        t_end = t_start + timedelta(seconds=2)

        """ Load Seismometer Data: """
        string_list = os.listdir("data/test_data/accumulation_seismometer/")
        if id == 5:
            filtered_strings = [s for s in string_list if s.startswith("ID:5_")]
        else:
            filtered_strings = [s for s in string_list if s.startswith("ID:"+str(id))]

        seis_data_path = "data/test_data/accumulation_seismometer/" + filtered_strings[0]
        seis_stream = read(seis_data_path, starttime=UTCDateTime(t_start),
                           endtime=UTCDateTime(t_end))
        seis_data = seis_stream[0].data
        seis_stats = seis_stream[0].stats
        seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
        seis_data = normalize_by_max(seis_data)


        """ Load DAS Data: """
        print(denoised_path)
        raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end, raw=True, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])
        denoised_data, denoised_headers, denoised_axis = load_das_data(denoised_path, t_start, t_end, raw=False, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])

        print("Denoised Data Shape: ", denoised_data.shape)

        """ Normalize Data for Plotting Reasons: """
        raw_data_norm = normalize_by_max(raw_data)
        denoised_data_norm = normalize_by_max(denoised_data)

        """ Calculate Residuals: """
        raw_minus_denoised_data = raw_data_norm - denoised_data_norm
        raw_minus_denoised_data_norm = normalize_by_max(raw_minus_denoised_data)

        """ Parameters for Plotting """
        cmap1 = "plasma"
        cmap2 = "Greys"
        t_start_das = 0
        t_end_das = denoised_data_norm.shape[1]
        ch_start = 0
        ch_end = denoised_data_norm.shape[0]
        channels = raw_data_norm.shape[0]
        middle_channel = event_times[id][1]
        ch_ch_spacing = 12
        vmin=-0.4
        vmax=0.4
        vmin_residual = -1
        vmax_residual = 1

        """ Plotting Denoised Data: """
        im1 = axs[i, 0].imshow(denoised_data_norm, cmap=cmap1, aspect="auto", interpolation="antialiased",
                  extent=(0 ,(t_end_das-t_start_das)/400,0,ch_end * ch_ch_spacing/1000),
                  vmin=vmin, vmax=vmax)

        """ Plotting Residuals: """
        im2 = axs[i, 1].imshow(raw_minus_denoised_data_norm, cmap=cmap2, aspect="auto", interpolation="antialiased",
                         extent=(0, (t_end_das - t_start_das) / 400, 0, ch_end * ch_ch_spacing / 1000),
                         vmin=vmin_residual, vmax=vmax_residual)

        """ Plotting Wiggle comparison """
        col_pink = "#CE4A75"
        col_dark_blue = "#11045E"
        if id == 82:
            t_start_wiggle = 320
            t_end_wiggle = 480
        else:
            t_start_wiggle = 270
            t_end_wiggle = 430

        axs[i, 2].plot(seis_data[t_start_wiggle:t_end_wiggle],
                       color=col_pink, label="Co-Located Seismometer", linewidth=1.5, alpha=1, zorder=1)

        axs[i, 2].plot(raw_data_norm[middle_channel][t_start_wiggle:t_end_wiggle],
                       color=col_dark_blue, label="Noisy", linewidth=1.2, alpha=0.4, zorder=1)

        axs[i, 2].plot(denoised_data_norm[middle_channel][t_start_wiggle:t_end_wiggle],
                       color=col_dark_blue, label="Denoised", linewidth=1.2, alpha=1, zorder=1)

        axs[i, 2].set_yticks([])
        ax2 = axs[i, 2].twinx()

        ax2.set_ylabel("Amplitude [norm.]", fontsize=fs, color="black")
        ax2.set_yticks([])
        ax2.tick_params(axis="y", labelcolor="red")
        #axs[i, 2].legend(fontsize=20)

        "Set Axes"

        axs[i, 0].set_ylabel(experiment_names[i], fontsize=fs)
        axs[i, 0].set_yticklabels([])
        axs[i, 1].set_yticklabels([])

        if id == 5:
            axs[i, 0].set_yticklabels([])
            axs[i, 1].set_yticklabels([])
            #axs[i, 0].tick_params(axis='y', labelsize=fs - 2)

        if id == 20:
            axs[i, 0].set_yticklabels([])
            axs[i, 1].set_yticklabels([])
            #axs[i, 0].tick_params(axis='y', labelsize=fs - 2)

        if id == 82:
            axs[i, 0].set_yticklabels([])
            axs[i, 1].set_yticklabels([])
            #axs[i, 0].tick_params(axis='y', labelsize=fs - 2)


        axs[i, 0].set_xticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 2].set_xticklabels([])

        axs[i, 2].set_xticklabels([])
        axs[i, 2].set_xticks([40, 80, 120])

        if i==5:
            axs[i, 0].set_xlabel("Time [s]", fontsize=fs)
            axs[i, 1].set_xlabel("Time [s]", fontsize=fs)
            axs[i, 2].set_xlabel("Time [s]", fontsize=fs)

            axs[i, 0].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)
            axs[i, 1].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)

            axs[i, 2].set_xticklabels([0.1, 0.2, 0.3], fontsize=fs-2)
            axs[i, 2].set_xlabel("Time [s]", fontsize=fs)


        """ Plot Arrow """
        arrow_style = "fancy,head_width=0.5,head_length=1.0"
        axs[i, 0].annotate("", xy=(0, (channels - middle_channel) * 0.0125),
                           xytext=(-0.05, (channels - middle_channel) * 0.0125),
                           arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))

        # plot arrow in time domain:
        marker_position_1 = t_start_wiggle / 400
        marker_position_2 = t_end_wiggle / 400
        if i == 5:
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

    """ Set Titles """
    axs[0, 0].set_title("Denoised", y=1.0, fontsize=fs+2)
    axs[0, 1].set_title("Residuals", y=1.0, fontsize=fs+2)
    axs[0, 2].set_title("Time Series Comparison", y=1.0, fontsize=fs+2)


    """ Add letters in plots """
    letter_params = {
        "fontsize": fs + 2,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w"}
    }
    letters = ["Aa", "Ab", "Ac", "Ba", "Bb", "Bc", "Ca", "Cb", "Cc", "Da", "Db", "Dc", "Ea", "Eb", "Ec", "Fa", "Fb", "Fc",
               "Ga", "Gb", "Gc", "Ha", "Hb", "Hc"]

    for i in range(len(experiments)):
        for j in range(3):
            axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i * 3 + j], **letter_params)

    """ Save Plot """
    plt.tight_layout()
    plt.show()
    #if id==5:
    #    plt.savefig("plots/figS7.pdf", dpi=400)
    #if id == 20:
    #    plt.savefig("plots/fig5.pdf", dpi=400)
    #if id==82:
    #    plt.savefig("plots/figS8.pdf", dpi=400)
