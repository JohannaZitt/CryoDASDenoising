from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import butter_bandpass_filter as bandpass_filter
from helper_functions import resample_DAS as resample
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5




"""

Here we generate the initial DAS data sections for model fine-tuning as described in Section 3.3 Model Fine-Tuning.
The waveforms are downsampled in space, downsampled in time, bandpass filtered, demeaned and normalized by std.


"""


########################################
### Event Times and Cable Positions: ###
########################################

# event_data, event_time, start_channel, end_channel
event01 = ["2020/07/07", "13:24:41.5", 1320, 1450] #Ablation Zone
event02 = ["2020/07/11", "12:34:15.0", 1260, 1310] #Ablation Zone
event03 = ["2020/07/12", "06:00:37.5", 1255, 1410] #Ablation Zone
event04 = ["2020/07/12", "06:53:13.5", 1160, 1215] #Ablation Zone
event05 = ["2020/07/12", "06:53:13.5", 1215, 1350] #Ablation Zone
event06 = ["2020/07/12", "06:53:28.0", 1200, 1350] #Ablation Zone
event07 = ["2020/07/13", "00:48:36.7", 395, 440] #Ablation Zone
event08 = ["2020/07/06", "19:32:37.5", 3425, 3550] #Accumulation Zone
event09 = ["2020/07/06", "20:42:37.5", 3410, 3510] #Accumulation Zone
event10 = ["2020/07/06", "20:41:49.0", 3400, 3520] #Accumulation Zone
event11 = ["2020/07/06", "19:27:10.5", 3410, 3510] #Accumulation Zone
event12 = ["2020/07/06", "20:19:00.5", 3420, 3550] #Accumulation Zone
event13 = ["2020/07/06", "19:42:27.0", 3400, 3510] #Accumulation Zone
event14 = ["2020/07/06", "19:51:23.0", 3400, 3510] #Accumulation Zone
events = [event01, event02, event03, event04, event05, event06, event07, event08, event09, event10, event11, event12, event13, event14]

###################
### Parameters: ###
###################

n_sub: int = 15
n_t: int = 2400
fs: int = 400

training_data = np.zeros((60, n_t, n_sub))
n: int = 0 # denotes the current saving spot in the training_data np.array

file_dir = "data/training_data/raw_DAS/"
savedir = "data/training_data/preprocessed_DAS/retraining_data.npy"


for event in events:

    print("Processing Event " , event)

    for m in range(3): # m represents the initial channel used for spatial downsampling of the data.

        #####################
        ### Reading Data: ###
        #####################

        event_time = datetime.strptime(event[0] + " " + event[1], "%Y/%m/%d %H:%M:%S.%f")
        t_start = event_time - timedelta(seconds=3)
        t_end   = t_start + timedelta(seconds=6)
        data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=file_dir, convert = False)

        ########################
        ### Processing Data: ###
        ########################

        # 1. downsample to 12 m channel spacing
        if data.shape[1] == 4864 or data.shape[1] == 4800:
            data = data[:, m::6]
            startchannel = event[2] // 6
            endchannel = event[3] // 6
        else:
            data = data[:, m::3]
            startchannel = event[2] // 3
            endchannel = event[3] // 3
        data = data[:, startchannel:endchannel]


        # 2. downsample data in time
        data = resample(data, headers["fs"] / fs)
        data = data[:n_t, :]

        # 3. filtering data
        for i in range(data.shape[0]):
            data[i] = bandpass_filter(data[i], 1, 120, 400, order=4)

        # 4. demean
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            data[:, i] = data[:, i] - mean

        # 5. scale by std
        for i in range(data.shape[1]):
            data[:,i] /= data[:,i].std()

        # 6. concatenate all data in one dataset training_data
        n_samples = int(data.shape[1] / n_sub)
        for i in range(n_samples):
            training_data[n] = data[:, i*n_sub:(i+1)*n_sub]
            n += 1


#################################
### Saving Initial Waveforms: ###
#################################
training_data = np.transpose(training_data, (0, 2, 1))
np.save(savedir, training_data)



######################
### Plotting Data: ###
######################

#def plot_data(data, moveout=8):
#    i = 0
#    alpha = 0.8
#    for ch in range(data.shape[0]):
#        plt.plot(data[ch][:] + moveout * i, '-k', alpha=alpha)
#        i += 1

#    plt.show()

#for j in range(training_data.shape[0]):
#    plot_data = training_data[j]
#    i: int  = 0
#    for ch in range(15):
#        plt.plot(plot_data[ch][:] + 10 * i, "-k", alpha=0.8)
#        i += 1
#    plt.show()




