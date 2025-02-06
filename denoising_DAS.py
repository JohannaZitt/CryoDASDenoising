import gc
import os
import sys
import time
from datetime import timedelta

import h5py
import keras
import numpy as np

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5
from helper_functions import butter_bandpass_filter




def resample(data, ratio):
    # spatial axis:
    n_ch = data.shape[0]
    if n_ch == 4800 or n_ch == 4864 or n_ch == 4928:
        data = data[::6, :]
    else:
        data = data[::3, :]

    # time axis
    res = np.zeros((data.shape[0], int(data.shape[1] / ratio)))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def denoise_file (file, timesamples, model, N_sub, fs_trainingdata):

    """

    Denoises a .h5 file of the Rhonegletscher data set.

    :param file: .h5 file to denoise
    :param timesamples: int: input size for the denoiser in time domain
    :param model: keras.model: model to use for denoising
    :param N_sub: int: input size for the denoiser in space domain
    :param fs_trainingdata: int: sample frequency of the training data
    :return: numpy array

    """

    """ Load data and headers """
    with h5py.File(file, "r") as f:
        DAS_data = f["Acoustic"][:]
    headers = load_das_h5.load_headers_only(file)

    """ Reshape data """
    DAS_data = DAS_data.T
    DAS_data = DAS_data.astype("f")
    DAS_data = resample(DAS_data, headers["fs"] / 400)

    """ Preprocessing data """
    for i in range(DAS_data.shape[0]):
        DAS_data[i] = butter_bandpass_filter(DAS_data[i], 1, 120, fs=400, order=4)
        DAS_data[i] = DAS_data[i] / np.std(DAS_data[i])

    """ Split data in input size for the denoiser """
    n_ch, n_t = DAS_data.shape
    n_samples = int(n_t / timesamples) + 1
    data = np.zeros((n_samples, n_ch, timesamples))
    for i in range(n_samples - 1):
        data[i] = DAS_data[:, i * timesamples: (i + 1) * timesamples]
    data[n_samples - 1] = DAS_data[:, n_t - timesamples: n_t]

    DAS_reconstructions = np.zeros_like(data)
    N_samples, N_ch, N_t = data.shape

    """ Loop over every batche of 1024 sampling points"""
    for n, eval_sample in enumerate(data):

        """ Prepare sample and masks """
        masks = np.ones((N_ch, N_sub, N_t, 1))
        eval_samples = np.zeros_like(masks)

        gutter = N_sub // 2
        mid = N_sub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :, 0] = eval_sample[:N_sub]

        for i in range(gutter, N_ch - gutter):
            start = i - mid
            stop = i + mid + 1

            masks[i, mid] = 0
            eval_samples[i, :, :, 0] = eval_sample[start:stop]

        for i in range(N_ch - gutter, N_ch):
            masks[i, i - N_ch] = 0
            eval_samples[i, :, :, 0] = eval_sample[-N_sub:]

        """ Demean """
        for i in range(eval_samples.shape[0]):
            for j in range(eval_samples.shape[1]):
                eval_samples[i, j, :] = eval_samples[i, j, :] - np.mean(eval_samples[i, j, :])


        """ Denoise Data """
        results = model.predict((eval_samples, masks))
        DAS_reconstructions[n] = np.sum(results, axis=1)[:, :, 0]

    """ Reshape data into original shape """
    data_save = np.zeros_like(DAS_data)
    for i in range(data_save.shape[0]):
        for j in range(DAS_reconstructions.shape[0] - 1):
            data_save[i, j * timesamples: (j + 1) * timesamples] = DAS_reconstructions[j, i, :]
    z = int(data_save.shape[1] / timesamples)
    x = z * timesamples
    y = data_save.shape[1] - x
    data_save[:, x: data_save.shape[1]] = DAS_reconstructions[DAS_reconstructions.shape[0] - 1, :, timesamples - y: timesamples]
    data_save = data_save.T

    """ Update Headers """
    headers["npts"] = data_save.shape[0]
    headers["nchan"] = data_save.shape[1]
    headers["fs"] = fs_trainingdata
    headers["dx"] = 12

    return data_save, headers

def deal_with_artifacts(data, filler = 0, Nt=1024):

    """

    Minimizing denoising artifacts

    :param data: numpy array
    :param filler: replacement value for intersections between two denoised batches
    :param Nt: batch size in time domain
    :return: numpy array
    """

    n_edges = int(data.shape[0] / Nt)
    for i in range(n_edges):
        for j in range(data.shape[1]):
            for n in range(3):
                data[Nt * i + n, j] = filler
                data[Nt * i - (n + 1), j] = filler

    return data


"""

Here we denoise the data as described in Section 3.4 Denoising Procedure


"""


models_path = "experiments"

# TODO Comment out to calculate values for all experiments
#model_names = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical",
#               "05_combined200", "06_combined800", "07_retrained_combined200", "08_retrained_combined800", "09_borehole_seismometer"]
model_names = ["03_accumulation_horizontal"]

raw_das_folder_path = "data/raw_DAS"

""" Parameters: """
n_sub = 11
timesamples = 1024
fs_trainingdata = 400
DEAL_WITH_ARTIFACTS = True

""" For every model: """
for model_name in model_names:

    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")
    print("#####################" + model_name + "##################################")
    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")

    """ Saving directory and Model directory: """
    saving_path = os.path.join("experiments", model_name, "denoisedDAS")
    model_file = os.path.join("experiments", model_name, model_name + ".h5")
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    files = os.listdir(raw_das_folder_path)

    print_count = 0

    """ For every file: """
    for file in files:
        print("------------ File: ", file, " is denoising --------------------")

        """ Load Data: """
        raw_das_file_path = os.path.join(raw_das_folder_path, file)
        print(raw_das_file_path)
        saving_filename = "/denoised_" + file

        """ Loas Model """
        model = keras.models.load_model(model_file)

        """ Denoise Data """
        if os.path.exists(saving_path+saving_filename):
            start = time.time()
            data, headers = denoise_file(file=raw_das_file_path, timesamples=timesamples, model=model, N_sub=n_sub, fs_trainingdata=fs_trainingdata)
            end = time.time()
            """ Deal with artifacts: """
            if DEAL_WITH_ARTIFACTS:
                print("deals with artifacts")
                data = deal_with_artifacts(data)

            """ Save Data: """
            write_das_h5.write_block(data, headers, saving_path + saving_filename)
            print("Saved Data with Shape: ", data.shape)

            if print_count < 50:

                print_count += 1

                """ Measuring Denoising Time """
                dur = end - start
                dur_str = str(timedelta(seconds=dur))
                x = dur_str.split(':')

                # print in file
                with open("experiments/" + model_name + "/" + model_name[:2] + "_runtimes.txt", "a") as w_file:
                    w_file.write(str(print_count) + ": File " + str(file) + " took " + str(x[1]) + " minutes and " + str(x[2]) + " seconds\n")


        else:
            print(saving_filename, " does exist already. No Denoising is performed. ")

        gc.collect()
