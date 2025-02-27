
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


""" load raw DAS data """

def plot_sectionplot(raw_data):

    """

    Plots data as waveform section plot

    """

    """ Parameters """
    font_m = 14
    channels = raw_data.shape[0]
    alpha = 0.3

    n = 0
    for ch in range(channels):
        plt.plot(raw_data[-ch][:] + 1.5 * n, "-k", alpha=alpha)
        n += 1
    plt.xticks([], [])
    plt.ylabel("Offset [km]", size=font_m)

    plt.tight_layout()
    plt.show()




# ID : [starttime, start channel delta, end channel delta, category, closts seismometer]
events = {5: ["2020-07-27 19:43:30.5", 1550, 2150, 1, "ALH", "5_"],
         20: ["2020-07-27 00:21:46.3", 1650, 2250, 2, "ALH", "20"],
         82: ["2020-07-27 05:04:55.0", 1705, 2305, 3, "ALH", "82"]
         }
ids = [5, 20, 82]
receiver = "ALH"

for id in ids:
    event_time = events[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_start = t_start - timedelta(seconds=2)
    t_end = t_start + timedelta(seconds=6)


    raw_folder_path = "data/raw_DAS/"
    raw_data, raw_headers, raw_axis = load_das_h5.load_das_custom(t_start,
                                                                  t_end,
                                                                  input_dir=raw_folder_path,
                                                                  convert=False)

    raw_data = raw_data[:, events[id][1]:events[id][2]]
    np.save("experiments/15_DASDL/denoisedDAS_image/npy_raw/raw_" + str(id) + ".npy", raw_data)

    for i in range(raw_data.shape[1]):
        raw_data[:, i] = raw_data[:, i] / np.abs(raw_data[:, i]).max()

    np.save("experiments/15_DASDL/denoisedDAS_image/npy_raw/raw_"+str(id)+".npy", raw_data)
    plot_sectionplot(raw_data.T)
    print(raw_data.shape)
    print(type(raw_data))
