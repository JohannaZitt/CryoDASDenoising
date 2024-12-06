import numpy as np
import scipy.signal



def fk_spectra(data, shift=True):
    """Function that calculates frequency-wavenumber spectrum of input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """

    nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.fft2(data))
        f_axis = np.fft.fftshift(np.fft.fftfreq(nTi, d=1/1000))
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=4))

    else:
        data_fk = np.fft.fft2(data)
        f_axis = np.fft.fftfreq(nTi, d=1/1000)
        k_axis = np.fft.fftfreq(nCh, d=4)

    return data_fk, f_axis, k_axis


def fkr_spectra(data, shift=True):
    """Function that calculates frequency-wavenumber spectrum of input data.

    Taking advantage that input signal is real to only compute posivite
    frequencies.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """

    nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.rfft2(data, axes=(1,0)), axes=1)
        f_axis = np.fft.rfftfreq(nTi, d=1/1000)
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=4))

    else:
        data_fk = np.fft.rfft2(data, axes=(1,0))
        f_axis = np.fft.rfftfreq(nTi, d=1/1000)
        k_axis = np.fft.fftfreq(nCh, d=4)

    return data_fk, f_axis, k_axis


def create_mask_BP(fk_data, freq_axis, k_axis, k_slope = 10, corner_freqs=[1, 100, -0.1, 0.1]):
    """
    Function to create filter mask for 2D bandpass filter
    :type corner_freqs: list
    :param corner_freqs: List giving corner frequencies for 2D bandpass filter as [fmin, fmax, kmin, kmax]
    """
    f_min, f_max, k_min, k_max = corner_freqs

    # Create empty mask of shape fk_data
    mask = np.zeros(fk_data.shape)

    # Get indices corresponding to corner frequencies
    cand_fmin = np.where(freq_axis<f_min)
    index_fmin = max(cand_fmin[0])

    cand_fmax = np.where(freq_axis>f_max)
    index_fmax = min(cand_fmax[0])

    cand_kmin = np.where(k_axis<k_min)
    index_kmin = max(cand_kmin[0])

    cand_kmax = np.where(k_axis>k_max)
    index_kmax = min(cand_kmax[0])

#     # specify slope of roll-off based on corner frequency
#     f_slope = f_max/6
#     k_slope = k_max/6

    # Set all elements of mask which are inside corners to 1
    mask[index_fmin:index_fmax, index_kmin:index_kmax] = 1

    # Smooth roll-off at edges to avoid edge effect of ifft2
    for i in range(k_slope):
        for j in range(index_kmin,index_kmax):
            mask[i,j] = i * (1/k_slope)

    for i in range(100):
        for j in range(index_kmin,index_kmax):
            mask[index_fmax+i,j] = 1 - i * (1/100)

    for i in range(index_fmin,index_fmax):
        for j in range(k_slope):
            mask[i,index_kmin-k_slope+j] = j * (1/k_slope)
            mask[i,index_kmax+j] = 1 - j * (1/k_slope)

    # Corner are chosen to contain the lower values coresponding to basic roll-off
    # right upper corner
    for i in range(index_fmax, index_fmax+100):
        for j in range(index_kmax, index_kmax+k_slope):
            freq_val = 1 - (i-index_fmax) * (1/100)
            k_val = 1 - (j-index_kmax) * (1/k_slope)
            if freq_val > k_val:
                mask[i,j] = k_val
            else:
                mask[i,j] = freq_val

    # left upper corner
    for i in range(index_fmax, index_fmax+100):
        for j in range(index_kmin-k_slope,index_kmin):
            freq_val = 1 - (i-index_fmax) * (1/100)
            k_val = (j-(index_kmin-k_slope)) * (1/k_slope)
            if freq_val > k_val:
                mask[i,j] = k_val
            else:
                mask[i,j] = freq_val

    # left lower corner
    for i in range(k_slope):
        for j in range(index_kmin-k_slope,index_kmin):
            freq_val = i * (1/k_slope)
            k_val = (j-(index_kmin-k_slope)) * (1/k_slope)
            if freq_val > k_val:
                mask[i,j] = k_val
            else:
                mask[i,j] = freq_val

    # right lower corner
    for i in range(k_slope):
        for j in range(index_kmax,index_kmax+k_slope):
            freq_val = i * (1/k_slope)
            k_val = 1 - (j-index_kmax) * (1/k_slope)
            if freq_val > k_val:
                mask[i,j] = k_val
            else:
                mask[i,j] = freq_val

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
    """
    fk_filtered = fk_data * fk_mask
    # Unshift k-axis
    fk_ishift = np.fft.ifftshift(fk_filtered, axes=1)

    # Transform data back to x-t domain
    data_recon = np.fft.irfft2(fk_ishift, axes=(1,0))

    return data_recon


# """
# The following code has been copied from the distpy module.
# https://github.com/Schlumberger/distpy/tree/master/distpy
# Access: April 5, 2021
# """
# def bounded_select(x, vel_low, vel_high):
#     nx=x.shape[0]
#     nt=x.shape[1]
#     velm = np.zeros((nx,nt),dtype=np.double)
#     velm[np.where(x > vel_low)]=1.0
#     velm[np.where(x > vel_high)]=0.0
#     return velm

# def vel_map(nx,nt,dx,dt):
#     # COULD BE GPU COMPATIBLE...BUT VEL_MASK IS NOT
#     TOLERANCE=1.0e-7
#     freq = np.linspace(0,1.0/(2*dt),nt//2)
#     wavn = np.linspace(0,1.0/(2*dx),nx//2)
#     frequency = np.zeros((1,nt),dtype=np.double)
#     frequency[0,:nt//2]=freq
#     frequency[0,-(freq.shape[0]):]=np.flipud(freq)
#     halfLength = nx//2
#     wavenumber = np.zeros((nx,1),dtype=np.double)
#     wavenumber[0:halfLength]=np.reshape(wavn,(halfLength,1))
#     lineTest = np.reshape(np.flipud(wavn),(halfLength,1))
#     wavenumber[halfLength:]= lineTest[:len(wavenumber[halfLength:])]
#     freqImage = np.reshape(np.tile(frequency,(nx,1)),(nx,nt))
#     wavnImage = np.reshape(np.tile(wavenumber,(1,nt)),(nx,nt))
#     return freqImage/(wavnImage+TOLERANCE)

# '''
# vel_mask - use the results of vel_map to create a filter that band passes phase velocities
#         - between vel_low and vel_high. The smoothing is applied so that the edges are not too
#         - sharp. The padtype='even' prevents strong filter edge effects.
# '''
# def vel_mask(velmap,vel_low,vel_high,smooth=0):
#     # NOT GPU COMPATIBLE DUE TO scipy.signal.filtfilt
#     nx=velmap.shape[0]
#     nt=velmap.shape[1]
#     velm = bounded_select(velmap, vel_low, vel_high)
#     if smooth>1:
#         Wn = 1.0/smooth
#         b, a = scipy.signal.butter(5,Wn,'low',output='ba')
#         velm = np.abs(scipy.signal.filtfilt(b,a,velm,axis=1, padtype='even'))
#     return velm


# def spectral_whiten()

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
    TOLERANCE=1.0e-7  # Avoid dividing by zero
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



def AGC(data_in, window, normalize=False, time_axis='vertical'):
    """
    Function to apply Automatic Gain Control (AGC) to seismic data.

    Parameters
    ----------
    data_in : numpy array or obspy Stream object
        The seismic ata.
    window : int
        window length in number of samples (not time).
    time_axis : string, optional
        confirm whether the input data has the time axis on the vertical or
        horizontal axis. Obspy Streams use horizontal axis by default. The default is 'vertical'.

    Returns
    -------
    y : numpy array or obspy stream object (dependent on input data_in)
        The data with AGC applied.

    tosum : AGC scaling values applied

    """
    # data: input data as numpy array, time is on the vertical axis, channel number on the horizontal axis
    # window: AGC window length in number of samples (not time)
#     stream = 0
#     if type(data_in) != np.ndarray:
#         stream = 1
#         data_np = np.zeros((len(data_in), len(data_in[0])))
#         data_st = data_in.copy()
#         for i in range(len(data_in)):
#             data_np[i] = data_in[i].data
#         data_in = data_np.copy()

    if time_axis != 'vertical':
        data_in = data_in.T

    nt = data_in.shape[0]
    nx = data_in.shape[1]

    y = np.zeros((nt, nx))
    tosum = np.zeros((nt, nx))

#     if type(data_in) != np.ndarray:
#         n, m = data_np.shape
#     else:
#         n, m = data_in.shape

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

#     if type(data_in) != np.ndarray:
#         e = np.cumsum(abs(data_np), axis=0)
#     else:
#         e = np.cumsum(abs(data_in), axis=0)
    e = np.cumsum(abs(data_in), axis=0)

    for i in range(nt):
        if i-len2-1 > 0 and i+len2 < nt:
            tosum[i,:] = e[i+len2,:] - e[i-len2-1]
        elif i-len2-1 < 1:
            tosum[i,:] = e[i+len2,:]
        else:
            tosum[i,:] = e[-1,:] - e[i-len2-1, :]

    for i in range(len2):
        tosum[i,:] = tosum[len2+1,:]
        tosum[-1-i,:] = tosum[-1-len2,:]

    y = data_in / tosum

    if normalize:
        y = y/np.max(abs(y))

#     if stream == 1:
#         for i in range(len(data_st)):
#             data_st[i].data = y.T[i]

#         y = data_st.copy()


    return y, tosum
