import os

import keras
import numpy as np



def denoise_file (DAS_data, model, timesamples = 1024, N_sub = 11):

    """

    Denoises synthetically generated DAS data.

    :param DAS_data: numpy array
    :param model: keras,model
    :param timesamples: int: input size for the denoiser in time domain
    :param N_sub: int: input size for the denoiser in space domain
    :return: numpy array

    """

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
        for i in range(eval_samples.shape[0]): # for every channel
            for j in range(eval_samples.shape[1]): # for every N_sub
                eval_samples[i, j, :] = eval_samples[i, j, :] - np.mean(eval_samples[i, j, :])

        """ Denoise Data """
        results = model.predict((eval_samples, masks))
        DAS_reconstructions[n] = np.sum(results, axis=1)[:, :, 0]

    """ Reshape data into original shape """
    data_save = np.zeros_like(DAS_data)
    for i in range(data_save.shape[0]):  # per channel
        for j in range(DAS_reconstructions.shape[0] - 1):  # per sample
            data_save[i, j * timesamples: (j + 1) * timesamples] = DAS_reconstructions[j, i, :]
    z = int(data_save.shape[1] / timesamples)
    x = z * timesamples
    y = data_save.shape[1] - x
    data_save[:, x: data_save.shape[1]] = DAS_reconstructions[DAS_reconstructions.shape[0] - 1, :, timesamples - y: timesamples]

    return data_save


"""

Here we denoise the synthetically generated test data as described in Section 3.4 Denoising Procedure


"""

# TODO Comment out to calculate values for all experiments
#model_names = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical",
#               "05_combined200", "06_combined800", "09_borehole_seismometer"]

model_names = ["01_ablation_horizontal"]

# TODO change to "from_seis" to denoise synthetic data generated from seismometer data
# TODO change to "from_DAS" to denoise synthetic data generated from DAS data
data_type = "from_DAS"

""" path to data """
data_path = "data/synthetic_DAS/" + data_type + "/"
data_files = os.listdir(data_path)

for model_name in model_names:

    print("##############################################################")
    print("##############################################################")
    print("############### " + model_name + " ####################")
    print("##############################################################")
    print("##############################################################")

    """ Load model """
    model = keras.models.load_model("experiments/" + model_name + "/" + model_name + ".h5")

    for data_file in data_files:

        """ Load Data """
        raw_data = np.load(data_path+data_file)

        """ Denoise Data """
        denoised_data = denoise_file(raw_data, model)

        saving_path = "experiments/" + model_name + "/denoised_" + data_path[5:]
        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        """ Save Data """
        np.save(saving_path + "/denoised_" + data_file.split("/")[-1], denoised_data)






