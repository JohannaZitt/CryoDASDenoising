"""
This script provides functions for signal processing, including calculating
the frequency-wavenumber spectrum and applying tapering to 2D data.

The functions included are:

1. fk_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
              zero_padding=True):
    - Calculates the frequency-wavenumber spectrum of input data.

2. fkr_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
               zero_padding=True):
    - Calculates the frequency-wavenumber spectrum of real input data,
      computing only positive frequencies.

3. taper_array_along_axis(data, axis='vertical', edge_length=None):
    - Tapers 2D data along a chosen axis. Can be called twice to taper along
      both axes.

The script is designed to assist in processing seismic or other time-series
data by transforming them to the frequency-wavenumber domain and preparing
the signal through tapering.
"""

import numpy as np
from scipy.signal.windows import hann


def fk_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
               zero_padding=True):
    """
    Calculate the frequency-wavenumber spectrum of input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain.
    fs : int
        Sampling frequency in Hz.
    dx : int
        Channel offset in meters.
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.
    zero_padding : bool, optional
        Zero-pad signal to next power of two before FFT. Defaults to True.

    Returns
    -------
    data_fk : numpy.ndarray
        f-k spectrum of input dataset.
    f_axis : numpy.ndarray
        Corresponding frequency axis.
    k_axis : numpy.ndarray
        Corresponding wavenumber axis.
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh = int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.fft2(data, s=(nTi, nCh)))
        f_axis = np.fft.fftshift(np.fft.fftfreq(nTi, d=1/fs))
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))
    else:
        data_fk = np.fft.fft2(data, s=(nTi, nCh))
        f_axis = np.fft.fftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def fkr_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
                zero_padding=True):
    """
    Calculate the frequency-wavenumber spectrum of real input data.

    Takes advantage of the fact that the input signal is real to only compute
    positive frequencies.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain.
    fs : int
        Sampling frequency in Hz.
    dx : int
        Channel offset in meters.
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.
    zero_padding : bool, optional
        Zero-pad signal to next power of two before FFT. Defaults to True.

    Returns
    -------
    data_fk : numpy.ndarray
        f-k spectrum of input dataset.
    f_axis : numpy.ndarray
        Corresponding frequency axis.
    k_axis : numpy.ndarray
        Corresponding wavenumber axis.
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh = int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.rfft2(data, s=(nTi, nCh),
                                               axes=(1, 0)), axes=1)
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))
    else:
        data_fk = np.fft.rfft2(data, s=(nTi, nCh), axes=(1, 0))
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def taper_array_along_axis(data, axis='vertical', edge_length=None):
    """
    Taper 2D data along a chosen axis. Call the function twice with different
    axis parameters to taper along both axes.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
    axis : str, optional
        Specify axis along which data is tapered.
    edge_length : int, optional
        Length of the edges which are tapered. Defaults to half the data
        points along the given axis.

    Returns
    -------
    numpy.ndarray
        Tapered data.
    """
    if axis == 'vertical':
        taper_multiplier = np.ones(data.shape[0])
        if not edge_length:
            edge_length = data.shape[0] // 4
        if edge_length % 2 == 0:
            edge_length += 1
        taper_window = hann(2 * edge_length)
        taper_multiplier[:edge_length] = taper_window[:edge_length]
        taper_multiplier[-edge_length:] = taper_window[-edge_length:]
        data_tapered = np.array([trace * taper_multiplier for trace in
                                 data.T]).T
    elif axis == 'horizontal':
        taper_multiplier = np.ones(data.shape[1])
        if not edge_length:
            edge_length = data.shape[1] // 2
        if edge_length % 2 == 0:
            edge_length += 1
        taper_window = hann(2 * edge_length)
        taper_multiplier[:edge_length] = taper_window[:edge_length]
        taper_multiplier[-edge_length:] = taper_window[-edge_length:]
        data_tapered = np.array([timepoint * taper_multiplier for timepoint
                                 in data])
    else:
        raise ValueError("Please define axis as 'vertical' or 'horizontal'")

    return data_tapered
