import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter

from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = "band")
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y

def resample_seis(data, ratio) -> NDArray:

    """
    :param data: np array
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def resample_DAS(data, ratio) -> NDArray:
    """
    :param data: np array
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    data = data.T
    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio)+1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res.T

def resample(data, ratio):
    try:
        res = np.zeros((int(data.shape[0]/ratio) + 1, data.shape[1]))
        for i in range(data.shape[1]):

            res[:,i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:,i])
    except ValueError as e:
        res = np.zeros((int(data.shape[0] / ratio), data.shape[1]))
        for i in range(data.shape[1]):
            res[:, i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:, i])
    return res

def xcorr(x, y):  # Code by Martijn van den Ende

    # FFT of x and conjugation
    X_bar = np.fft.rfft(x).conj()
    Y = np.fft.rfft(y)

    # Compute norm of data
    norm_x_sq = np.sum(x ** 2)
    norm_y_sq = np.sum(y ** 2)
    norm = np.sqrt(norm_x_sq * norm_y_sq)

    # Correlation coefficients
    R = np.fft.irfft(X_bar * Y) / norm

    # Return correlation coefficient
    return np.max(R)


def compute_xcorr_window(x):  # Code by Martijn van den Ende
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan

    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])

    return np.nanmean(Cxy)


def compute_moving_coherence(data, bin_size):  # Code by Martijn van den Ende

    N_ch = data.shape[0]

    cc = np.zeros(N_ch)

    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])

    return cc

def get_middel_channel(receiver):
    channel = 0
    if receiver == "AKU":
        channel = 3740
    elif receiver == "AJP":
        channel = 3460
    elif receiver == "ALH":
        channel = 3842
    elif receiver == "RA82":
        channel = 1300
    elif receiver == "RA85":
        channel = 1070
    elif receiver == "RA87":
        channel = 1230
    elif receiver == "RA88":
        channel = 1615 # 1600
    else:
        print("There is no start nor end channel for receiver " + receiver + '.')

    channel = int(channel/6)
    return channel


def load_das_data(folder_path, t_start, t_end, raw, channel_delta_start, channel_delta_end):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype('f')

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers['dx'] = 12

    # 3. cut to size
    ch_middel = int(3842/6)
    data = data[:, ch_middel - channel_delta_start:ch_middel + channel_delta_end]

    if raw:
        # 4. downsample in time
        data = resample(data, headers['fs'] / 400)
        headers['fs'] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
        data[:,i] = data[:,i] / np.std(data[:,i])
        #data[:, i] = data[:, i] / np.abs(data[:, i]).max()

    return data.T, headers, axis