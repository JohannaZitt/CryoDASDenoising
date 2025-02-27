
import numpy as np

def compute_snr(trace, metric="power"):
    #print("trace shape: ", trace.shape)
    max_position = np.argmax(trace)
    signal_window_length = 200 # correspondes to 0.5 seconds recoring

    cut_position = int(np.clip(max_position, signal_window_length / 2, trace.shape[0] - signal_window_length / 2))
    start_signal = int(cut_position - signal_window_length / 2)
    end_signal = int(cut_position + signal_window_length / 2 + 1)

    short_window = trace[start_signal:end_signal]
    long_window = np.concatenate((trace[:start_signal], trace[end_signal:]), axis=0)
    #print("short_window_length: ", short_window.shape)
    #print("long_window_length: ", long_window.shape)

    if metric == "power":
        value_signal = np.mean(short_window ** 2, axis=0)
        value_noise = np.mean(long_window ** 2, axis=0)
    elif metric == "variance":
        value_signal = np.var(short_window, axis=0)
        value_noise = np.var(long_window, axis=0)
    elif metric == "rms":
        value_signal = np.sqrt(np.mean(short_window ** 2, axis=0))
        value_noise = np.sqrt(np.mean(long_window ** 2, axis=0))
    elif metric == "absolute":
        value_signal = np.mean(np.absolute(short_window), axis=0)
        value_noise = np.mean(np.absolute(long_window), axis=0)
    snr = value_signal / value_noise

    return snr

data_folders = ["data/training_data/preprocessed_seismometer/01_ablation_horizontal.npy",
                "data/training_data/preprocessed_seismometer/03_accumulation_horizontal.npy",
                "data/training_data/preprocessed_seismometer/05_combined200.npy"]

for data_folder in data_folders:
    data = np.load(data_folder)

    #print(data)

    snr_values = []
    for trace in data:
        snr_value = compute_snr(trace, metric="absolute") #variance, power, rms, absolute
        snr_values.append(snr_value)
    print("\n")
    print("Data Type: ", data_folder[44:])
    print("Length: ", len(snr_values))
    print("SNR value: ", np.mean(snr_values))
