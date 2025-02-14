import matplotlib.pyplot as plt
import numpy as np

from helper_functions import compute_moving_coherence


"""

Here Figure 2 is generated


"""


def normalize_by_max(data):
    abs_max = np.max(np.abs(data))
    return data / abs_max

def normalize_minmax(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

""" Parameters """

cmap = "plasma"
vmin = -0.3
vmax = 0.3
ch_start = 10
ch_end = 70
ch_total = 60

t_start = 750
t_end = t_start + 800

t_start_wiggle = 1040
t_end_wiggle = t_start_wiggle + 200
channel_wiggle_comparison = 28#28

SNR_values = [0.0, 1.0, 3.2, 10.0]

col_pink = "#CE4A75"
fs=16 # fontsize
delta = 2 # fontsize delta

noisy_data_path = "data/synthetic_DAS/from_seis/"
noisy_data_names = ["clean_ID:46_SNR:0.npy", "ID:46_SNR:10.0.npy",  "ID:46_SNR:3.2.npy", "ID:46_SNR:1.0.npy"]
denoised_data_path = "experiments/02_accumulation_horizontal/denoised_synthetic_DAS/from_seis/"


""" Load Ground Truth Data"""
ground_truth_data = np.load(noisy_data_path + noisy_data_names[0])[ch_start:ch_end]
ground_truth_data = normalize_by_max(ground_truth_data)
print(ground_truth_data.shape)

""" Create Plot """
fig, axs = plt.subplots(4, 4, figsize=(12, 14), gridspec_kw={"width_ratios": [5, 5, 1, 5]})

for i, noisy_data_name in enumerate(noisy_data_names):

    """ Load Data """
    data = np.load(noisy_data_path + noisy_data_name)[ch_start:ch_end]
    data = normalize_by_max(data)
    denoised_data = np.load(denoised_data_path + "denoised_" + noisy_data_name)[ch_start:ch_end]
    denoised_data = normalize_by_max(denoised_data)

    print(data.shape)
    print(denoised_data.shape)

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
    axs[i, 0].imshow(data[:, t_start:t_end], cmap=cmap, aspect="auto", interpolation="antialiased",
          vmin=vmin, vmax=vmax)
    axs[i, 1].imshow(denoised_data[:, t_start:t_end], cmap=cmap, aspect="auto", interpolation="antialiased",
                     vmin=vmin, vmax=vmax)
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color = "black")
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")

    """ Plotting wiggle for wiggle comparison """
    axs[i, 3].plot(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color=col_pink,
                       label="Target Waveform", linewidth=1.5, alpha=0.8, zorder=1)

    if not i == 0:
        axs[i, 3].plot(0.8*data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey",
                   label="Noisy", linewidth=1.5, alpha=0.6, zorder=1)

    axs[i, 3].plot(0.6*denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
               label="Denoised", linewidth=1.5, alpha=0.8, zorder=1)



    #if not i == 0:
    #    axs[i, 3].plot(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey", label="Synthetics",
    #                   linewidth=1.5, alpha=0.6, zorder=1)
    #else:
    #    axs[i, 3].plot(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
    #               label="Denoised", linewidth=1.5, alpha=0.8, zorder=1)

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

    axs[i, 0].set_xticks([200, 400, 600], [0.5, 1, 1.5], fontsize = fs-delta)
    axs[i, 1].set_xticks([200, 400, 600], [0.5, 1, 1.5], fontsize = fs-delta)
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


axs[3, 0].annotate("", xy=(t_start_wiggle-t_start, 59.5),
                        xytext=(t_start_wiggle-t_start, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_start_wiggle-t_start, 59.5),
                        xytext=(t_start_wiggle-t_start, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 0].annotate("", xy=(t_start_wiggle-t_start+200, 59.5),
                        xytext=(t_start_wiggle-t_start+200, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_start_wiggle-t_start+200, 59.5),
                        xytext=(t_start_wiggle-t_start+200, 59.9),
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









