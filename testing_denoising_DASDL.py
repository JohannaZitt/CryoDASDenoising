import os
import numpy as np
import h5py
from datetime import datetime, timedelta
from helper_functions import load_das_data, butter_bandpass_filter
import matplotlib.pyplot as plt
from pyDASDL.bp import bandpass
from tensorflow import keras
from pyDASDL.Utils import patch, patch_inv
from pyDASDL.fk import *
from pyDASDL.cwt_2d import cwt_2d

""" Event IDs"""
# specify event ID: [event time, start_channel, amount_channel, category, receiver]
event_times = {
    82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH"],
               }
id = 82


raw_path = os.path.join("data", "raw_DAS/")

event_time = event_times[id][0]
t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
t_end = t_start + timedelta(seconds=2)

""" Load DAS Data: """
raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end, raw=True, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])


""" Plot Raw Data: """

cmap = "plasma"
t_start_das = 0
t_end_das = raw_data.shape[1]
ch_start = 0
ch_end = raw_data.shape[0]
channels = raw_data.shape[0]
middle_channel = event_times[id][1]
ch_ch_spacing = 12
vmin=-7
vmax=7

plt.imshow(raw_data.T, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end_das-t_start_das)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=vmin, vmax=vmax)
plt.show()

""" Denoise Data: """
f = h5py.File(r'experiments/15_DASDL/matlab_output/rohnegltscher_sample900_2000_PreProcessed.mat')
#BP = np.array(np.transpose(f.get('outF')))
CWTSCALE = np.array(np.transpose(f.get('out')))
dn = np.array(np.transpose(f.get('dn')))


#BP = bandpass(dn, 0.001, 0.001, 120, 6, 6, 0)

# Mein BP funktioniert!!!
BP = dn
for i in range(dn.shape[1]):
    BP[:, i] = butter_bandpass_filter(BP[:, i], 1, 120, 1000, 4)



#ss_log = np.arange(1, 11, 0.5)
#coeffs_nsq, wav_norm_nsq = cwt_2d(dn, ss_log, 'mexh')
#CWTSCALE = np.abs(coeffs_nsq[:, :, - 1])




""" With Own BP Filter: """
print("RAW SHAPE ", raw_data.shape)
print("DN SHAPE: ", dn.shape)
print("BP SHAPE: ", BP.shape)
print("CWTSCALE SHAPE: ", CWTSCALE.shape)


# params
w1=48
w2=48
s1z=8
s2z=8

model = keras.models.load_model("experiments/15_DASDL/DASDL_models/DAS_PATCH_Johanna.h5")
#model.summary()

# """ Patching the CWT SCALE """
cwt_scale_patch = patch(CWTSCALE, w1, w2, s1z, s2z)
cwt_scale_input = np.reshape(cwt_scale_patch, (cwt_scale_patch.shape[0], w1 * w2, 1))

# """ Patching the Band-pass Filtered Data """
band_pass_patch = patch(BP, w1, w2, s1z, s2z)
band_pass_input = np.reshape(band_pass_patch, (band_pass_patch.shape[0], w1 * w2, 1))

# """ Denoising Data """
out = model.predict([band_pass_input, cwt_scale_input], batch_size=32)

out = np.reshape(out,(out.shape[0],w1*w2))
outA = np.transpose(out)
n1,n2=np.shape(BP)
outB = patch_inv(outA,n1,n2,w1,w2,s1z,s2z)
denoised_data = np.array(outB)

denoised_data_dip = denoised_data - fkdip(denoised_data,0.02)


plt.imshow(denoised_data_dip, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end_das-t_start_das)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=-1e5, vmax=1e5)
plt.show()





