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

""" Event IDs"""
# specify event ID: [event time, start_channel, amount_channel, category, receiver]
event_times = {0: ["2020-07-27 08:17:34.5", 40, 40, 1, "ALH"],
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"],

               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"],

               82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH"],
               }

""" Experiment"""
experiment = "15_DASDL" # "01_ablation_horizontal", "03_accumulation_horizontal", "07_retrained_combined200", 11_vanende, 12_vanende_finetuned_cryo, 13_isken_filter, 14_julius_filter, 15_DASDL

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
    seis_data = seis_data / np.std(seis_data)


    """ Load DAS Data: """
    #raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end, raw=True, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])
    #denoised_data, denoised_headers, denoised_axis = load_das_data(denoised_path, t_start, t_end, raw=False, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])

    import h5py

    file_name_denoised = "experiments/15_DASDL/denoisedDAS/denoised_DASDL_rhone1khz_UTC_20200727_194308.575.h5"
    file_name_raw = "data/raw_DAS/rhone1khz_UTC_20200727_194308.575.h5"

    # Datei öffnen
    with h5py.File(file_name_raw, 'r') as f:
        # Direktes Laden des Acoustic Datasets
        acoustic_data = f['Acoustic'][:]

        # Lass uns mal die Form/Dimensionen anschauen
        print("Shape:", acoustic_data.shape)
        # Und den Datentyp
        print("Dtype:", acoustic_data.dtype)

        # Datei öffnen
    with h5py.File(file_name_denoised, 'r') as f:
        # In die Acquisition Gruppe schauen
        acq_group = f['Acquisition']

        # Zeigen wir mal was in dieser Gruppe ist
        print(list(acq_group.keys()))