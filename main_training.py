import os
import time
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import random as python_random
import tensorflow as tf

from models import UNet, CallBacks, DataGenerator
from models import seed


"""

Primary training is conducted on the initial waveforms generated using seismometer data.

The code is built upon the software provided by van den Endet et al. [1].

[1] van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A. ve Richard, C. (2021, 3 Mart). A Self-Supervised 
Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data. 
figshare. doi:10.6084/m9.figshare.14152277.v1

"""

# NumPy (random number generator used for sampling operations)
tf.random.set_seed(seed)
python_random.seed(seed)
rng = np.random.default_rng(seed)
tf.config.threading.set_inter_op_parallelism_threads(4)

""" Parameters """
N_sub = 11
batch_size = 32
Nt = 1024
N_epoch = 2000
# set batch multiplier to 10 for 200 initial waveforms and to 3 for 800 training samples
batch_multiplier = 10

model_params = {
    "use_bn": False, # batch normalization
    "use_dropout": False,
    "dropout_rate": 0.1,
    "N_blocks": 4, # amount of downsampling and upsampling blocks
    "f0": 4, # dimension of output space (i.e. the number of output filters in the convolution)
    "LR": 1e-4, # learning rate
    "data_shape": (N_sub, Nt, 1),
    "kernel": (3,5),
    "AA": True # anti aliasing
}


different_training_data = ["01_ablation_horizontal"]#, "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical", "05_combined200", "09_borehole_seismometer"] # set batch multiplier to 10
#different_training_data = ["06_combined800"] # set batch multiplier to 3

for training_data in different_training_data:

    """ Saving Paths """
    cwd = os.getcwd()
    savedir = os.path.join("experiments", training_data)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    """ Saving Path log """
    logdir = os.path.join("experiments", training_data, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    """ Callbacks """
    tensorboard_callback = CallBacks.tensorboard(logdir)
    checkpoint_callback = CallBacks.checkpoint(os.path.join(savedir, training_data + ".h5"))

    """ Trainingdata """
    data_file = os.path.join(cwd, "data", "training_data", "preprocessed_seismometer", training_data + ".npy")

    """ Load data """
    data = np.load(data_file)
    N_ch, N_t = data.shape

    """ Split data 80-20 train-test """
    split = int(0.8 * N_ch)
    train_data = data[:split]
    test_data = data[split:]

    print("Preparing masks")
    train_generator = DataGenerator(X=train_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    test_generator = DataGenerator(X=test_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    print("Done")

    """ Visualize training data:  """
    #for i in range(10):
    #    for j in range(11):
    #        plt.plot(train_generator.samples[i][j] + 12*j, color="black", alpha=0.5)
    #    plt.show()


    """ Construct model """
    net = UNet()
    net.set_params(model_params)
    model = net.construct()
    # model.summary()

    """ Model training """
    start = time.time()
    print("Start Model Training")
    model.fit(
        x=train_generator,
        validation_data=test_generator,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=1, epochs=N_epoch,
    )

    """ Measure Runtime """
    end = time.time()
    dur = end-start
    dur_str = str(timedelta(seconds=dur))
    x = dur_str.split(":")
    output_text = training_data + ": " + str(x[0]) + " hours, " + str(x[1]) + " minutes and " + str(x[2]) + " seconds."

    print(output_text)
