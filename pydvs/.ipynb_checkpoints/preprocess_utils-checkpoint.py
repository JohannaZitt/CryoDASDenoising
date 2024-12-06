"""
This script provides a set of functions for signal processing, including
Automatic Gain Control (AGC), tapering, spectral whitening, and energy
normalization.

The functions included are:

1. bandpass_filter(data_array, freqmin, freqmax, fs, corners=4,
                   dec_factor=None):
    - Applies a bandpass filter to channels in a 2D array with
      zero phase shift.

2. AGC_1D(input_trace, window_length):
    - Applies Automatic Gain Control (AGC) to a single trace.

3. AGC(data_in, window, normalize=False, time_axis='vertical'):
    - Applies Automatic Gain Control (AGC) to seismic data.

4. create_edge_taper(num_samples, max_percentage=0.05):
    - Creates a time-domain tapering window where only the edges are tapered.

5. taper(data, max_percentage=0.25, ddim=2):
    - Applies tapering to a 1D or 2D data array.

6. double_whiten(input_trace, freq_smooth_window=10, time_smooth_window=100):
    - Applies spectral whitening followed by temporal normalization.

7. single_whiten_taper(input_trace, freq_taper, freq_smooth_window=10):
    - Applies spectral whitening to a trace and then tapers in the
      frequency domain.

8. energy_norm(data_matrix):
    - Normalizes the energy of each column in a 2D data array.

The script is designed to assist in processing seismic or other time-series
data by normalizing and enhancing the signal through various methods of gain
control, tapering, and whitening.
"""

import numpy as np
from scipy.signal.windows import hann


def bandpass_filter(data_array, freqmin, freqmax, fs, corners=4,
                    dec_factor=None):
    from scipy.signal import iirfilter, zpk2sos, sosfilt
    """
    Apply a bandpass filter to each seismic trace in a 2D array with
    zero phase shift.

    Parameters
    ----------
    data_array : numpy.ndarray
        2D array representing seismic traces with time along the first axis
        and space along the second axis.
    freqmin : float
        Low cutoff frequency of the bandpass filter (Hz).
    freqmax : float
        High cutoff frequency of the bandpass filter (Hz).
    fs : float
        Sampling frequency (Hz).
    corners : int, optional
        Order of the Butterworth filter. Default is 4.
    dec_factor : int, optional
        Decimation factor. If provided, the data will be decimated by this
        factor.

    Returns
    -------
    filtered_data : numpy.ndarray
        Bandpass filtered seismic traces.
    """
    nyquist = 0.5 * fs
    low = freqmin / nyquist
    high = freqmax / nyquist

    # Design a butterworth bandpass filter
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    firstpass = sosfilt(sos, data_array, axis=0)
    data_filtered = sosfilt(sos, firstpass[::-1, :], axis=0)[::-1, :]

    if dec_factor:
        if dec_factor >= (fs / (2*freqmax)):
            print("""'dec_factor' too large for given 'freqmax'!
                  There might be aliasing.""")
        return data_filtered[::dec_factor, :]
    else:
        return data_filtered


def AGC_1D(input_trace, window_length):
    """
    Apply Automatic Gain Control (AGC) to a single trace.

    Parameters
    ----------
    input_trace : 1D numpy array
        Input data.
    window_length : int
        Window length in number of samples (not time).

    Returns
    -------
    normalized_trace : 1D numpy array
        Amplitude normalized input trace.
    agc_scaling_values : 1D numpy array
        AGC scaling values that were applied.
    """
    num_samples = len(input_trace)
    normalized_trace = np.zeros(num_samples)
    agc_scaling_values = np.zeros(num_samples)

    # Ensure that window_length is an integer
    if not isinstance(window_length, int):
        window_length = int(window_length)

    # Ensure that window_length is smaller than the length of the time series
    if window_length > num_samples:
        window_length = num_samples

    # Ensure that window_length is odd
    if window_length % 2 == 0:
        window_length -= 1

    half_window = (window_length - 1) // 2

    cumulative_sum = np.cumsum(np.abs(input_trace))

    for i in range(num_samples):
        if i - half_window - 1 > 0 and i + half_window < num_samples:
            agc_scaling_values[i] = cumulative_sum[i + half_window] - \
                                    cumulative_sum[i - half_window - 1]
        elif i - half_window - 1 < 1:
            agc_scaling_values[i] = cumulative_sum[i + half_window]
        else:
            agc_scaling_values[i] = cumulative_sum[-1] - \
                                    cumulative_sum[i - half_window - 1]

    normalized_trace = input_trace / agc_scaling_values
    return normalized_trace, agc_scaling_values


def AGC(data_in, window_length, normalize=False, time_axis='vertical'):
    """
    Apply Automatic Gain Control (AGC) to seismic data.

    Parameters
    ----------
    data_in : numpy.ndarray
        Input data.
    window_length : int
        Window length in number of samples (not time).
    normalize : bool, optional
        If True, normalize the output. Default is False.
    time_axis : str, optional
        Confirm whether the input data has the time axis on the vertical or
        horizontal axis. Default is 'vertical'.

    Returns
    -------
    agc_data : numpy.ndarray
        Data with AGC applied.
    agc_scaling_values : numpy.ndarray
        AGC scaling values that were applied.
    """
    if time_axis != 'vertical':
        data_in = data_in.T

    num_samples, num_traces = data_in.shape
    agc_data = np.zeros((num_samples, num_traces))
    agc_scaling_values = np.zeros((num_samples, num_traces))

    # Ensure that window_length is an integer
    if not isinstance(window_length, int):
        window_length = int(window_length)

    # Ensure that window_length is smaller than the length of the time series
    if window_length > num_samples:
        window_length = num_samples

    # Ensure that window_length is odd
    if window_length % 2 == 0:
        window_length -= 1

    half_window = (window_length - 1) // 2

    cumulative_sum = np.cumsum(np.abs(data_in), axis=0)

    for i in range(num_samples):
        if i - half_window - 1 > 0 and i + half_window < num_samples:
            agc_scaling_values[i, :] = cumulative_sum[i + half_window, :] - \
                                       cumulative_sum[i - half_window - 1, :]
        elif i - half_window - 1 < 1:
            agc_scaling_values[i, :] = cumulative_sum[i + half_window, :]
        else:
            agc_scaling_values[i, :] = cumulative_sum[-1, :] - \
                                       cumulative_sum[i - half_window - 1, :]

    for i in range(num_traces):
        agc_data[:, i] = data_in[:, i] / agc_scaling_values[:, i]

    if normalize:
        agc_data /= np.max(np.abs(agc_data))

    return agc_data, agc_scaling_values


def create_edge_taper(num_samples, max_percentage=0.05):
    """
    Create a tapering window where only the edges are tapered.

    Parameters
    ----------
    num_samples : int
        Total length of the tapering window.
    max_percentage : float, optional
        Maximum percentage of the total length to be tapered at each edge.
        Default is 0.05.

    Returns
    -------
    taper_window : 1D numpy array
        Tapering window.
    """
    taper_window = np.ones(num_samples)
    edge_length = int(max_percentage * num_samples)

    # # Ensure that edge_length is odd
    # if edge_length % 2 == 0:
    #     edge_length += 1

    # Create Hann window, split it, and set values to 1 in between
    hann_window = hann(2 * edge_length + 1)
    taper_window[:edge_length] = hann_window[:edge_length]
    taper_window[-edge_length:] = hann_window[-edge_length:]
    return taper_window


def taper(data, max_percentage=0.25, ddim=2):
    """
    Apply tapering to a 1D or 2D data array.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
    max_percentage : float, optional
        Length of the taper edge. If not provided, 50% of input data length is
        used. Default is 0.25.
    ddim : int, optional
        Dimension of data. Default is 2.

    Returns
    -------
    tapered_data : numpy.ndarray
        Tapered data.
    """
    num_samples = len(data)
    taper_window = create_edge_taper(num_samples, max_percentage)

    if ddim == 1:
        tapered_data = data * taper_window
    elif ddim == 2:
        tapered_data = data * taper_window.reshape(-1, 1)
    return tapered_data


def double_whiten(input_trace, freq_smooth_window=10, time_smooth_window=100):
    """
    Apply spectral whitening followed by temporal normalization.

    Parameters
    ----------
    input_trace : 1D numpy array
        Input data.
    freq_smooth_window : int
        Window length in number of samples for smoothing in the frequency
        domain.
    time_smooth_window : int
        Window length in number of samples for smoothing the trace in the time
        domain.

    Returns
    -------
    double_normalized_trace : 1D numpy array
        Spectrally whitened and time normalized trace.
    """
    # Perform spectral whitening
    spectrum = np.fft.rfft(input_trace)
    spectrum_normalized, _ = AGC_1D(spectrum, freq_smooth_window)
    whitened_trace = np.fft.irfft(spectrum_normalized)

    # Perform temporal normalization
    double_normalized_trace, _ = AGC_1D(whitened_trace, time_smooth_window)
    return double_normalized_trace


def single_whiten_taper(input_trace, freq_taper, freq_smooth_window=10):
    """
    Apply spectral whitening to a trace and then taper in the frequency domain.

    Parameters
    ----------
    input_trace : 1D numpy array
        Input data.
    freq_taper : 1D numpy array
        Frequency representation of the bandpass filter to be applied after
        spectral whitening.
    freq_smooth_window : int
        Window length in number of samples for smoothing in the frequency
        domain.

    Returns
    -------
    whitened_tapered_trace : 1D numpy array
        Spectrally whitened and tapered trace.
    """
    # Perform spectral whitening
    spectrum = np.fft.rfft(input_trace)
    spectrum_whitened, _ = AGC_1D(spectrum, freq_smooth_window)

    # Apply the frequency taper
    spectrum_whitened_tapered = spectrum_whitened * freq_taper
    whitened_tapered_trace = np.fft.irfft(spectrum_whitened_tapered)
    return whitened_tapered_trace


def energy_norm(data_matrix):
    """
    Normalize the energy of each column in a 2D data array.

    Parameters
    ----------
    data_matrix : 2D numpy array
        Input data matrix.

    Returns
    -------
    normalized_matrix : 2D numpy array
        Energy-normalized data matrix.
    """
    energy_sum = (data_matrix ** 2).sum(axis=0)
    normalized_matrix = data_matrix / energy_sum.reshape(1, -1)
    return normalized_matrix
