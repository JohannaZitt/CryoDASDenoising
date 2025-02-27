import numpy as np
import matplotlib.pyplot as plt


def stream2array(stream, startCh=0, endCh=None):
    """
    Convert Stream Object to 2d numpy array

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Stream object containing seismic traces
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered

    Returns
    -------
    ensemble : 2d numpy array of input stream
    """
    if not endCh:
        endCh = len(stream)
    
    nStations = endCh - startCh
    nTimePoints = stream[startCh].stats.npts
    ensemble = np.zeros((nTimePoints, nStations))

    for channelNumber in range(startCh,endCh):
        ensemble[:,channelNumber-startCh] = stream[channelNumber].data
        
    return ensemble


def plotChRange(data,startCh=0,endCh=None,fs=1000, clipPerc=None, time_axis='vertical',
                cmap='seismic', dpi=200, title=None, outfile=None):
    """
    Plot DAS profile for given channel range
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered,  in case only part of the channels is
        of interest.
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T
    
    if not endCh:
        endCh = data.shape[1]
        
    nSec = float(data.shape[0]) / float(fs)
    if clipPerc:
        clip = np.percentile(np.absolute(data),clipPerc)
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-clip, vmax=clip, cmap=cmap,extent=(startCh,endCh-1,nSec,0))
    else:    
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-abs(data).max(), vmax=abs(data).max(),
                   cmap=cmap,extent=(startCh,endCh-1,nSec,0))
    plt.xlabel('Channel number')
    plt.ylabel('Time (s)')
    plt.colorbar(label='Strain rate ($10^{-9}$ $s^{-1}$)')
    if title:
        plt.title(title)
    if outfile:
        plt.savefig(outfile)
    plt.show()
    return


def fk_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical'):
    """
    Function that calculates frequency-wavenumber spectrum of input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T
    
    nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.fft2(data))
        f_axis = np.fft.fftshift(np.fft.fftfreq(nTi, d=1/fs))
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.fft2(data)
        f_axis = np.fft.fftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def fkr_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical'):
    """
    Function that calculates frequency-wavenumber spectrum of real input
    data.

    Taking advantage that input signal is real to only compute posivite
    frequencies.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T
    
    nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.rfft2(data, axes=(1,0)), axes=1)
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.rfft2(data, axes=(1,0))
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def plotFkSpectra(data_fk, f_axis, k_axis, log=False, vmax=None, dpi=200,
                  title=None, outfile=None):
    """
    Plot fk-spectra of data

    Parameters
    ----------
    data_fk : numpy.ndarray
        f-k spectrum of input dataset
    f_axis : numpy.ndarray
        corresponding frequency axis
    k_axis : numpy.ndarray
        corresponding wavenumber axis
    log : bool, optional
        If set True, amplitude of plot is logarithmic
    vmax : float, optional
        Set max value of colormap. If set to None, colormap is applied to min
        and max value of the data
    dpi : int, optional
        Resolution of figure
    title : str, optional
        Title of the figure. If set to None, no title is plotted.
    outfile : str, optional
        Path where to save figure. If set to None, figure is not saved.

    Returns
    -------
    Plot of f-k spectra
    """
    extent = (k_axis[0],k_axis[-1],f_axis[-1],f_axis[0])
    plt.figure(dpi=dpi)
    if log:
            plt.imshow(np.log10(abs(data_fk)), aspect='auto',
                       interpolation='none', extent=extent,
                       cmap='viridis', vmax=vmax)
    else:
        plt.imshow(abs(data_fk), aspect='auto', interpolation='none',
                   extent=extent, cmap='viridis', vmax=vmax)
    if title:
        plt.title(title)
    plt.xlabel('Wavenumber (1/m)')
    plt.ylabel('Frequency (1/s)')
    cbar = plt.colorbar()
    if log:
        cbar.set_label('$\mathrm{log}_{10}(\mathrm{Amplitude}^{2})$')
    else:
        cbar.set_label('$\mathrm{Amplitude}^{2}$')
    if outfile:
        plt.savefig(outfile)
    plt.show()
    return


def gaussian_decay(width, truncate=3.2):
    """Gaussian-like decay for tapering filter edges

    Parameters
    ----------
    width : int
        Desired with of roll-off region
    truncate : int, optional
        Number of standard diviations contained in the result. Defaults to 3.2,
        which corresponds to a lower value of the decay function of 0.005

    Returns
    -------
    Sorted array of length width, with Gaussian decay from 1 to 0
    """
    x = np.linspace(0, truncate, width)
    return np.exp(-(x*x) / 2)


def vel_map(nx,nt,dx,dt):
    """Function to create velocity map for given parameters of FK domain

    Parameters
    ----------
    nx : int
        Number of wavenumbers points
    nt : int
        Number of frequency points
    dx : int
        Spatial sampling rate
    dt : int
        Temporal sampling rate

    Returns
    -------
    vmap : Velocity map of given FK domain
    """
    TOLERANCE=1.0e-10  # Avoid dividing by zero
    frequency = np.fft.rfftfreq(nt,dt)
    wavenumber = np.fft.fftshift(np.fft.fftfreq(nx,dx))
    fImage = np.tile(frequency,(nx,1)).T
    kImage = np.tile(np.abs(wavenumber),(1,nt//2+1)).reshape(nt//2+1,nx)
    vmap = fImage/(kImage+TOLERANCE)
    return vmap


def vel_mask(velmap, vel_low, vel_high, smooth_perc=[-0.1, 0.1]):
    """
    Function to create a velocity mask for FK filtering.

    Parameters
    ----------
    velmap : numpy.ndarray
        Velocity map of the FK domain
    vel_low : int
        Lowest velocity for wich energy is kept.
    vel_high : int
        Highest velocity for wich energy is kept.
    smooth_perc : list, optional
        Gives the percentages to determine the roll-off range of the filter mask.
        E.g. -10% to +10% of cutoff-velocity are subject to Guassian decay

    Returns
    -------
    mask : The filter mask in the FK domain. Note that the mask only contains the
            positive frequencies.


    """
    # -10% to +10% of actuall cutoff velocity used as the roll-off region
    vel_low_l = vel_low * (1+smooth_perc[0])
    vel_low_h = vel_low * (1+smooth_perc[1])
    vel_high_l = vel_high * (1+smooth_perc[0])
    vel_high_h = vel_high * (1+smooth_perc[1])

    # Initialize filter mask
    nt=velmap.shape[0]
    nx=velmap.shape[1]
    mask = np.zeros((nt,nx))
    mask[np.where(velmap > vel_low_h)] = 1.0
    mask[np.where(velmap > vel_high_l)] = 0.0

    # Gaussian decay inside roll-off region, applied along frequency axis
    for k in range(nx):
        # Determine all elements lying inside lower roll-off region
        rolloff_low = np.where(np.logical_and(velmap[:,k]>=vel_low_l,
                                                     velmap[:,k]<=vel_low_h))[0]
        width_low = len(rolloff_low)
        gaussian_roll_l = gaussian_decay(width_low)
        mask[rolloff_low,k] = np.flipud(gaussian_roll_l)

        # Determine all elements lying inside upper roll-off region
        rolloff_high = np.where(np.logical_and(velmap[:,k]>=vel_high_l,
                                               velmap[:,k]<=vel_high_h))[0]

        width_high = len(rolloff_high)
        gaussian_roll_h = gaussian_decay(width_high)
        mask[rolloff_high,k] = gaussian_roll_h

    return mask


def apply_fk_mask(fk_data, fk_mask):
    """Apply filter mask to mute unwanted content in fk specturm and convert
    back to x-t domain using numpy.irfft2

    Parameters
    ----------
    fk_data : numpy.ndarray
        Frequency-wavenumber spectrum of data_fk
    fk_mask : numpy.ndarray
        Filter mask

    Returns
    -------
    data_recon : filtered data in t-x domain
    fk_filtered : muted f-k spectrum
    """
    # Compute magnitude and phase of fk spectra
    fk_mag = np.abs(fk_data)
    fk_phase = np.angle(fk_data)
    
    # Mute the magnitudes only, leave phase alone
    magn_muted = fk_mag * fk_mask

    # Combine muted magnitude and phase back together
    fk_filtered = magn_muted* np.exp(1j*fk_phase)
    
    # Unshift k-axis
    fk_ishift = np.fft.ifftshift(fk_filtered, axes=1)

    # Transform data back to x-t domain
    data_recon = np.fft.irfft2(fk_ishift, axes=(1,0))

    return data_recon, fk_filtered


def AGC(data_in, window, normalize=False, time_axis='vertical'):
    """
    Function to apply Automatic Gain Control (AGC) to seismic data.

    Parameters
    ----------
    data_in : numpy array
        Input data
    window : int
        window length in number of samples (not time)
    time_axis : string, optional
        Confirm whether the input data has the time axis on the vertical or
        horizontal axis

    Returns
    -------
    y : Data with AGC applied

    tosum : AGC scaling values that were applied

    """    
    if time_axis != 'vertical':
        data_in = data_in.T

    nt = data_in.shape[0]
    nx = data_in.shape[1]

    y = np.zeros((nt, nx))
    tosum = np.zeros((nt, nx))


    # enforce that window is integer
    if type(window) != int:
        window = int(window)

    # enforce that window is smaller than length of time series
    if window > nt:
        window = nt

    # enforce that window is odd
    if window % 2 == 0:
        window = window -1

    len2 = int((window-1)/2)

    e = np.cumsum(abs(data_in), axis=0)

    for i in range(nt):
        if i-len2-1 > 0 and i+len2 < nt:
            tosum[i,:] = e[i+len2,:] - e[i-len2-1,:]
        elif i-len2-1 < 1:
            tosum[i,:] = e[i+len2,:]
        else:
            tosum[i,:] = e[-1,:] - e[i-len2-1,:]

    for i in range(len2):
        tosum[i,:] = tosum[len2+1,:]
        tosum[-1-i,:] = tosum[-1-len2,:]

    y = data_in / tosum

    if normalize:
        y = y/np.max(abs(y))

    return y, tosum