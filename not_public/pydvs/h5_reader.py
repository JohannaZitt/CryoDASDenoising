"""
Simple script for reading native Silixa iDAS .h5 files. The module provides
functions to inspect metadata, read data from specified channels, and handle
multiple .h5 files, checking for data continuity and alignment.
"""

import pathlib
import warnings

import h5py
import numpy as np
from obspy import UTCDateTime


def peak_h5_idas2_data(file):
    """
    Inspects a native Silixa iDAS .h5 file and returns critical metadata.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the .h5 file.

    Returns
    -------
    starttime : UTCDateTime
        Start time of the data.
    endtime : UTCDateTime
        End time of the data.
    fs : int
        Sampling frequency in Hz.
    dx : float
        Spatial resolution in meters.
    d0 : float
        Start distance in meters.
    """
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['raw_das_data']
        ddims = dset.shape
        nsamp = ddims[0]
        metadata = dict(dset.attrs)
        starttime = UTCDateTime(str(metadata["starttime"]))
        fs = int(metadata["sampling_frequency_Hz"])
        dx = float(metadata["spatial_resolution_m"])
        d0 = float(metadata["start_distance_m"])
        endtime = starttime + ((nsamp - 1) / float(fs))

    return starttime, endtime, fs, dx, d0


def read_idas2_h5_file(file, channels=[0, -1]):
    """
    Reads data from specified channels of a single Silixa iDAS .h5 file.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the .h5 file.
    channels : list, optional
        List of channel indices to read. If [0, -1], all channels will be read.
        Defaults to [0, -1].

    Returns
    -------
    data : numpy.ndarray
        Data array.
    channels : list
        List of channels.
    metadata : dict
        Metadata dictionary.
    """
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['raw_das_data']
        metadata = dict(dset.attrs)

        if channels == [0, -1]:
            data = np.array(dset[:, :])
        elif len(channels) > 2:
            data = np.array(dset[:, channels])
        else:
            data = np.array(dset[:, channels[0]:channels[-1]])

        return data, channels, metadata


def das_reader(files, channels=[0, -1], debug=True):
    """
    Reads and processes multiple Silixa iDAS .h5 files, checking for data
    continuity and alignment.

    Parameters
    ----------
    files : list of str or str
        List of file paths or a single file path.
    channels : list, optional
        List of channel indices to read. Defaults to [0, -1].
    debug : bool, optional
        Flag to print debug information. Defaults to True.

    Returns
    -------
    data : numpy.ndarray
        Combined data array.
    channels : list
        List of channels.
    metadata : dict
        Metadata dictionary.
    """
    if not isinstance(files, list):
        files = [files]

    if debug:
        print('\U0001F50D Reading in: \n', files)

    # Check if consecutive files are well aligned, giving a warning if not
    for i in range(len(files) - 1):
        starttime0, endtime0, fs0, dx0, d00 = peak_h5_idas2_data(files[i])
        starttime1, endtime1, fs1, dx1, d01 = peak_h5_idas2_data(files[i + 1])

        if (fs0 == fs1) & (dx0 == dx1) & (d00 == d01):
            pass
        else:
            warnings.warn("Different acquisition parameters.", UserWarning)
            break

        if starttime1 == endtime0 + (1. / fs1):
            pass
        else:
            warnings.warn("Misaligned data (gaps or overlap).", UserWarning)
            break

    # Loop over all files and stack data into a single data array
    iter_ = (i for i in range(len(files)))
    for file in files:
        i = next(iter_)
        if i == 0:
            data, channels, metadata = read_idas2_h5_file(
                file, channels=channels)
        else:
            data_tmp, _, _ = read_idas2_h5_file(file, channels=channels)
            data = np.vstack([data, data_tmp])

    if debug:
        print("\U00002714 success")

    return data, channels, metadata
