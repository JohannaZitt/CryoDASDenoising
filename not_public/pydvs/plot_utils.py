"""
This script provides a set of functions for visualizing DAS data, including
converting an Obspy Stream object to a 2D numpy array, plotting data for a
given channel range, and plotting frequency-wavenumber (f-k) spectra.

The functions included are:

1. stream2array(stream, start_channel=0, end_channel=None):
    - Converts an ObsPy Stream object to a 2D numpy array.

2. plot_channel_range(data, start_channel=0, end_channel=None, fs=1000,
                      clip_percentage=None, time_axis='vertical',
                      cmap='seismic', dpi=200, title=None, outfile=None):
    - Plots a DAS profile for a specified range of channels.

3. plot_channel_range_ax(ax, data, start_channel=0, end_channel=None,
                         fs=1000, clip_percentage=None,
                         time_axis='vertical', cmap='seismic',
                         title=None):
    - Plots DAS profile in a specified axes (suitable for subplots)

4. plot_fk_spectra(data_fk, f_axis, k_axis, log_scale=False, vmax=None,
                   dpi=200, title=None, outfile=None):
    - Plots the frequency-wavenumber (f-k) spectra of seismic data.

The script is designed to assist in the visualization of
DAS data, making it easier to analyze and interpret the data.
"""

import numpy as np
import matplotlib.pyplot as plt


def stream_to_array(stream, start_ch=0, end_ch=None):
    """
    Convert a Stream object to a 2D numpy array.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Stream object containing seismic traces.
    start_ch : int, optional
        First channel considered, in case only part of the channels is of
        interest. Defaults to 0.
    end_ch : int, optional
        Last channel considered. If None, considers all channels.

    Returns
    -------
    ensemble : numpy.ndarray
        2D numpy array of the input stream.
    """
    if end_ch is None:
        end_ch = len(stream)

    num_stations = end_ch - start_ch
    num_time_points = stream[start_ch].stats.npts
    ensemble = np.zeros((num_time_points, num_stations))

    for channel in range(start_ch, end_ch):
        ensemble[:, channel - start_ch] = stream[channel].data

    return ensemble


def plot_channel_range(data, start_channel=0, end_channel=None, fs=1000,
                       clip_percentage=None, time_axis='vertical',
                       cmap='seismic', dpi=200, title=None, outfile=None):
    """
    Plot DAS profile for a given channel range.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain.
    start_channel : int, optional
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0.
    end_channel : int, optional
        Last channel considered, in case only part of the channels is
        of interest. If None, all channels are included.
    fs : int, optional
        Sampling frequency in Hz. Defaults to 1000.
    clip_percentage : float, optional
        Clip percentage for the plot. If None, no clipping is applied.
    time_axis : str, optional
        Axis along which time is represented ('vertical' or 'horizontal').
        Defaults to 'vertical'.
    cmap : str, optional
        Colormap for the plot. Defaults to 'seismic'.
    dpi : int, optional
        Resolution of the figure. Defaults to 200.
    title : str, optional
        Title of the figure. If None, no title is plotted.
    outfile : str, optional
        Path where to save the figure. If None, the figure is not saved.

    Returns
    -------
    None
    """
    if time_axis != 'vertical':
        data = data.T

    if end_channel is None:
        end_channel = data.shape[1]

    num_seconds = float(data.shape[0]) / float(fs)

    if clip_percentage is not None:
        clip_value = np.percentile(np.absolute(data), clip_percentage)
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none', vmin=-clip_value,
                   vmax=clip_value, cmap=cmap, extent=(start_channel,
                                                       end_channel - 1,
                                                       num_seconds, 0))
    else:
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-np.abs(data).max(), vmax=np.abs(data).max(),
                   cmap=cmap, extent=(start_channel, end_channel - 1,
                                      num_seconds, 0))

    plt.xlabel('Channel number')
    plt.ylabel('Time (s)')
    # plt.colorbar(label='Strain rate ($10^{-9}$ $s^{-1}$)')
    plt.colorbar(label='Strain rate (a. u.)')

    if title is not None:
        plt.title(title)

    if outfile is not None:
        plt.savefig(outfile)

    plt.show()


def plot_channel_range_ax(ax, data, start_channel=0, end_channel=None,
                          fs=1000, clip_percentage=None,
                          time_axis='vertical', cmap='seismic',
                          title=None):
    """
    Plot DAS profile for a given channel range on specified axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the image.
    data : numpy.ndarray
        Input data in the t-x domain.
    start_channel : int, optional
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0.
    end_channel : int, optional
        Last channel considered, in case only part of the channels is
        of interest. If None, all channels are included.
    fs : int, optional
        Sampling frequency in Hz. Defaults to 1000.
    clip_percentage : float, optional
        Clip percentage for the plot. If None, no clipping is applied.
    time_axis : str, optional
        Axis along which time is represented ('vertical' or 'horizontal').
        Defaults to 'vertical'.
    cmap : str, optional
        Colormap for the plot. Defaults to 'seismic'.
    title : str, optional
        Title of the figure. If None, no title is plotted.

    Returns
    -------
    None
    """
    if time_axis != 'vertical':
        data = data.T

    if end_channel is None:
        end_channel = data.shape[1]

    num_seconds = float(data.shape[0]) / float(fs)

    if clip_percentage is not None:
        clip_value = np.percentile(np.absolute(data), clip_percentage)
        im = ax.imshow(data, aspect='auto', interpolation='none',
                       vmin=-clip_value, vmax=clip_value, cmap=cmap,
                       extent=(start_channel, end_channel - 1,
                               num_seconds, 0))
    else:
        im = ax.imshow(data, aspect='auto', interpolation='none',
                       vmin=-np.abs(data).max(), vmax=np.abs(data).max(),
                       cmap=cmap, extent=(start_channel, end_channel - 1,
                                          num_seconds, 0))

    ax.set_xlabel('Channel number')
    ax.set_ylabel('Time (s)')
    ax.set_title(title if title else '')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Strain rate ($10^{-9}$ $s^{-1}$)')


def plot_fk_spectra(data_fk, f_axis, k_axis, log_scale=False, vmax=None,
                    dpi=200, title=None, outfile=None):
    """
    Plot f-k spectra of data.

    Parameters
    ----------
    data_fk : numpy.ndarray
        f-k spectrum of input dataset.
    f_axis : numpy.ndarray
        Corresponding frequency axis.
    k_axis : numpy.ndarray
        Corresponding wavenumber axis.
    log_scale : bool, optional
        If True, amplitude of plot is logarithmic. Defaults to False.
    vmax : float, optional
        Set max value of colormap. If None, colormap is applied to min
        and max value of the data. Defaults to None.
    dpi : int, optional
        Resolution of the figure. Defaults to 200.
    title : str, optional
        Title of the figure. If None, no title is plotted.
    outfile : str, optional
        Path where to save the figure. If None, the figure is not saved.

    Returns
    -------
    None
    """
    extent = (k_axis[0], k_axis[-1], f_axis[-1], f_axis[0])
    plt.figure(dpi=dpi)

    if log_scale:
        plt.imshow(np.log10(np.abs(data_fk) / np.abs(data_fk).max()),
                   aspect='auto', interpolation='none', extent=extent,
                   cmap='viridis', vmax=vmax)
    else:
        plt.imshow(np.abs(data_fk), aspect='auto', interpolation='none',
                   extent=extent, cmap='viridis', vmax=vmax)

    if title is not None:
        plt.title(title)

    plt.xlabel('Wavenumber (1/m)')
    plt.ylabel('Frequency (1/s)')
    colorbar = plt.colorbar()

    if log_scale:
        colorbar.set_label('Normalized Power [dB]')
    else:
        colorbar.set_label('PSD')

    if outfile is not None:
        plt.savefig(outfile)

    plt.show()
