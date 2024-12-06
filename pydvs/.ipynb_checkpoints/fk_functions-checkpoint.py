import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning

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
                   vmin=-clip, vmax=clip, cmap=cmap,extent=(startCh,endCh-1,
                                                            nSec,0))
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


def fk_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
               zero_padding=True):
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
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh= int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.fft2(data, s=(nTi,nCh)))
        f_axis = np.fft.fftshift(np.fft.fftfreq(nTi, d=1/fs))
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.fft2(data, s=(nTi,nCh))
        f_axis = np.fft.fftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def fkr_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
                zero_padding=True):
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
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh= int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.rfft2(data, s=(nTi,nCh), axes=(1,0)),
                                  axes=1)
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.rfft2(data, s=(nTi,nCh), axes=(1,0))
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
            plt.imshow(np.log10(abs(data_fk)/abs(data_fk).max()), aspect='auto',
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
        cbar.set_label('Normalized Power [dB]')
    else:
        cbar.set_label('PSD')
    if outfile:
        plt.savefig(outfile)
    plt.show()
    return


def taper_array_1D(data, axis='vertical', edge_length=None):
    """
    Function to taper 2D data along a chosen axis. Call the function twice
    with different axis parameters to taper along both axis.

    Parameters
    ----------
    data : numpy.ndarray
        Input data
    axis : str, optional
        Specify axis along which data is tapered.
    edge_length : int, optional
        Length of the edges which are tapered. Defaults to half the data
        points along given axis

    Returns
    -------
    Tapered data
    """
    if axis == 'vertical':
        taper_multiplier = np.ones(data.shape[0])
        # If no edge_length is given, taper 50% of input data
        if not edge_length:
            edge_length = data.shape[0] // 4
        # Make sure edge_length is odd
        if edge_length % 2 == 0:
            edge_length += 1
        # Create hanning window, split it and set values to 1 inbetween
        taper_window = hanning(2*edge_length)
        taper_multiplier[:edge_length] = taper_window[:edge_length]
        taper_multiplier[-edge_length:] = taper_window[-edge_length:]
        # Multiply each trace with the constructed taper window
        data_tapered = np.array([trace * taper_multiplier for trace in
                                 data.T]).T

    elif axis == 'horizontal':
        taper_multiplier = np.ones(data.shape[1])
        # If no edge_length is given, taper 50% of input data
        if not edge_length:
            edge_length = data.shape[1] // 2
        # Make sure edge_length is odd
        if edge_length % 2 == 0:
            edge_length += 1
        # Create hanning window, split it and set values to 1 inbetween
        taper_window = hanning(2*edge_length)
        taper_multiplier[:edge_length] = taper_window[:edge_length]
        taper_multiplier[-edge_length:] = taper_window[-edge_length:]
        # Multiply each time instance with the constructed taper window
        data_tapered = np.array([timepoint * taper_multiplier for timepoint
                                 in data])

    else:
        print("Please define axis as 'vertical' or 'horizontal'")

    return data_tapered



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


def vel_map_sym(nx,nt,dx,dt, zero_padding=True):
    """Function to create (symmetric) velocity map for given parameters of
    FK domain

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
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    vmap : Velocity map of given FK domain
    """
    # Extend nt and nx to next bigger power of two, thereby increasing speed
    # as well as zero-padding
    if zero_padding:
        nx = int(2**np.ceil(np.log2(nx)))
        nt = int(2**np.ceil(np.log2(nt)))
    TOLERANCE = 1e-7  # Avoid dividing by zero
    frequency = np.fft.rfftfreq(nt,dt)
    wavenumber = np.fft.fftshift(np.fft.fftfreq(nx,dx))
    fImage = np.tile(frequency,(nx,1)).T
    kImage = np.tile(np.abs(wavenumber),(1,nt//2+1)).reshape(
                     nt//2+1,nx)
    vmap = fImage/(kImage+TOLERANCE)
    return vmap


def vel_map(nt, nx, dt=0.002, dx=2, zero_padding=True):
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
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    vmap : Velocity map of given FK domain
    """
    # Extend nt and nx to next bigger power of two, thereby increasing speed
    # as well as zero-padding
    if zero_padding:
        nx = int(2**np.ceil(np.log2(nx)))
        nt = int(2**np.ceil(np.log2(nt)))

    TOLERANCE = 1e-7
    f_axis = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    k_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

    f_image = np.tile(f_axis, (nx,1)).T
    k_image = np.tile(k_axis, (1,nt)).reshape(nt,nx)

    return f_image / (k_image + TOLERANCE)


def Gauss_smooth(V, V1, Gsmooth=0.2):
    """Gaussian smoothing for tapering filter edges

    Parameters
    ----------
    V1 : numpy.ndarray
        Velocity map
    V1 : int
        Cutoff velocity
    Gsmooth : float, optional
        Variance for controlling smoothness

    Returns
    -------
    1d smoothed filter edge
    """
    Vdiff = V-V1
    return np.exp(-(Vdiff/V1)**2/Gsmooth**2)


def make_mask(vel_map, velmin, velmax, reject_or_retain='reject',
              Gsmooth_vmin=0.2, Gsmooth_vmax=0.2):
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
    reject_or_retain : str, optional
        Decide if energy between velmin and velmax is muted while everything
        else is kept ('reject') or if everything is muted except energy inside
        the velocity range ('retain')
    Gsmooth_vmin : float, optional
        Variance of Gaussian for controlling smoothness of lower filter edge
    Gsmooth_vmax : float, optional
        Variance of Gaussian for controlling smoothness of higher filter edge

    Returns
    -------
    mask : The filter mask in the FK domain.
    """
    mask = np.zeros(vel_map.shape)
    # Loop over all wavenumbers and smooth mask along time-axis
    for k in range(vel_map.shape[1]):
        vel_vector = vel_map[:,k]
        v_diff_min = vel_vector-velmin
        v_diff_max = vel_vector-velmax
        V2keep = np.ones(vel_map.shape[0])  # or to reject! ;)
        gaussian_min = Gauss_smooth(V=vel_vector, V1=velmin,
                                    Gsmooth=Gsmooth_vmin)
        gaussian_max = Gauss_smooth(V=vel_vector, V1=velmax,
                                    Gsmooth=Gsmooth_vmax)
        V2keep[v_diff_min<0] = gaussian_min[v_diff_min<0]
        V2keep[v_diff_max>0] = gaussian_max[v_diff_max>0]
        mask[:,k] = V2keep

    if reject_or_retain == 'reject':
        mask = 1 - mask
    return mask


def vel_mask_old(velmap, vel_low, vel_high, smooth_perc=[-0.1, 0.1]):
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
        Gives the percentages to determine the roll-off range of the filter
        mask. E.g. -10% to +10% of cutoff-velocity are subject to Guassian decay

    Returns
    -------
    mask : The filter mask in the FK domain. Note that the mask only contains
        the positive frequencies.
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


def apply_fk_mask(fk_data, fk_mask, nt, nx, rfft=False):
    """Apply filter mask to mute unwanted content in fk specturm and convert
    back to x-t domain using numpy.irfft2. Size of original data has to be
    specified in order to reverse effect of zero-padding.

    Parameters
    ----------
    fk_data : numpy.ndarray
        Frequency-wavenumber spectrum of data_fk
    fk_mask : numpy.ndarray
        Filter mask
    nt : int
        Number of time points of xt-data
    nx : int
        Number of channels of xt-data
    rfft : bool, optional
        Specifies if rfft has been used to compute spectra. Defaults to false

    Returns
    -------
    data_recon : filtered data in t-x domain
    fk_filtered : muted f-k spectrum
    """
    # It is not necessary to apply the commented lines. It's faster to directly
    # scale fk_spectrum with mask (phase information is retained either way)
    # # Compute magnitude and phase of fk spectra
    # fk_mag = np.abs(fk_data)
    # fk_phase = np.angle(fk_data)
    #
    # # Mute the magnitudes only, leave phase alone
    # magn_muted = fk_mag * fk_mask
    #
    # # Combine muted magnitude and phase back together
    # fk_filtered = magn_muted * np.exp(1j*fk_phase)

    # Apply the filter mask to the fk_spectrum
    fk_filtered = fk_data * fk_mask

    if rfft:
        # Unshift k-axis
        fk_ishift = np.fft.ifftshift(fk_filtered, axes=1)

        # Transform data back to x-t domain
        data_recon = np.fft.irfft2(fk_ishift, axes=(1,0))
    else:
        # Unshift k-axis
        fk_ishift = np.fft.ifftshift(fk_filtered)

        # Transform data back to x-t domain
        data_recon = np.fft.ifft2(fk_ishift)

    return data_recon[:nt,:nx], fk_filtered


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
