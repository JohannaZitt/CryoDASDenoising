# Filer of the paper "Denoising DAS data using an adaptive frequency-wavenumber filter"
# from Marius Isken et al. (2022)

"""

Miscellanous:
1. increasing the window size, the computation time also increases
2.


"""


from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from lightguide.blast import Blast

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from helper_functions import resample


""" LOAD DATA """
event_times = {#0: ["2020-07-27 08:17:34.5", 40, 40, 1, "ALH"],
               5: ["2020-07-27 19:43:30.5", 590, 700, 1, "ALH"],
               20: ["2020-07-27 00:21:46.3", 600, 670, 2, "ALH"],
               82: ["2020-07-27 05:04:55.0", 580, 770, 3, "ALH"],
               }

id = 82

raw_data_path = "/home/johanna/PycharmProjects/MAIN_DAS_denoising/data/raw_DAS/"

event_time = event_times[id][0]
t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
t_end = t_start + timedelta(seconds=2)
data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=raw_data_path, convert=False)
data = resample(data, 1000/400) # donwsample data from 1000 to 400 Hz
data = data[:, ::3] # downsample data to 12 m spacing
headers["fs"] = 400
headers["dx"] = 12
data = data.astype("f")



# Generate .mseed file for using lightguide:
st = Stream()
for i in range(data.shape[1]):
    trace_data = np.ascontiguousarray(data[:, i])
    trace = Trace(data=trace_data)
    trace.stats.sampling_rate = headers["fs"]
    trace.stats.starttime = t_start
    trace.stats.station = str(i)
    st.append(trace)
st.write("sample_output.mseed", format="MSEED", encoding = 4)


data = data.T

""" Plotting Params: """
vmin = -1
vmax = 1
cmap = "plasma"


#exponents = [0.2, 0.4, 0.6, 0.8, 1]
#window_sizes = [16, 32, 64, 128]
#overlaps = [1, 3, 5, 7]


#for exponent in exponents:
#    for window_size in window_sizes:
#        for overlap in overlaps:

# denoise data
blast = Blast.from_miniseed("sample_output.mseed")
blast.bandpass(max_freq=120, min_freq=1)
start = time.time()
blast.afk_filter(exponent=0.8, window_size=16, overlap=7, normalize_power=False)
denoised_data = blast.data
end = time.time()

""" Plotting Data: """
fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figheight(12)
fig.set_figwidth(24)

axs[0].imshow(data[event_times[id][1]:event_times[id][2]], cmap=cmap, aspect="auto", interpolation="antialiased")
axs[1].imshow(denoised_data[event_times[id][1]:event_times[id][2]], cmap=cmap, aspect="auto", interpolation="antialiased")


plt.show()
#plot_name = "exponent_" + str(exponent) + "_window_size_" + str(window_size) + "_overlap_" + str(overlap) + ".pdf"
#plt.savefig("filter_things/plots_iske_filter/" + plot_name)
#plt.close()


#dur = end-start
#dur_str = str(timedelta(seconds=dur))
#x = dur_str.split(":")
#print("Exponent: " + str(exponent) + "; Window_Size: " + str(window_size) + "; Overlap: " + str(overlap) + "; Time: " + str(x[2]))





