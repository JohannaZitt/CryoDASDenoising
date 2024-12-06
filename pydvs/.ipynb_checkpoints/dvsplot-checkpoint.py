import numpy as np
import matplotlib.pyplot as plt


def createEnsemble(stream, startCh, endCh):
    """
    Convert Stream Object to 2d numpy array
    :type data: obspy.core.stream.Stream
    :param data: Stream Object to convert.
    :param startCh: First channel considered.
    :param endCh: Last channel considered.
    :return: 2d numpy array of input.
    """
    nStations = endCh - startCh
    npts = stream[startCh].stats.npts
    ensemble = np.zeros((nStations,npts))

    for channelNumber in range(startCh,endCh):
        ensemble[channelNumber-startCh] = stream[channelNumber].data
        
    return ensemble


def plotChRange(data,startCh,endCh,samplesPerSec=1000, clipPerc=95, cmap='seismic', dpi=200, title=None, outfile=None):
    """Plot DAS profile for given channel Range"""
    clip= np.percentile(np.absolute(data),clipPerc)
    nSec = float(data.shape[1]) / float(samplesPerSec)
    plt.figure(dpi=dpi)
    plt.imshow(data,aspect='auto',interpolation='none',vmin=-clip,vmax=clip,
               cmap=cmap,extent=(0,nSec,endCh-1,startCh))
    plt.xlabel('Time (s)')
    plt.ylabel('Channel number')
#     plt.title('{} - {}'.format(st0[0].stats.starttime + 3.6, st0[0].stats.starttime + 4), fontsize=8)
    plt.colorbar(label='Strain rate ($10^{-9}$ $s^{-1}$)')
    
    if title:
        plt.title(title)
    if outfile:
        plt.savefig(outfile)
#         plt.savefig('report/3000deepQuake-ch-'+str(startCh)+'-'+str(endCh)+'.png')
#     plt.clf()
    plt.show()
    return


def plotFkSpectra(data_fk, f_axis, k_axis, log=False, vmax=None, dpi=200, title=None, outfile=None):
    """Plot fk spectra of data"""
    extent = (k_axis[0],k_axis[-1],f_axis[0],f_axis[-1])
    plt.figure(dpi=dpi)
    if log:
            plt.imshow(np.log10(abs(data_fk)), aspect='auto', interpolation='none',
                       extent=extent, cmap='viridis', vmax=vmax)
    else:
        plt.imshow(abs(data_fk), aspect='auto', interpolation='none',
                   extent=extent, cmap='viridis', vmax=vmax)
    if title:
        plt.title(title)
    plt.xlabel('Wavenumber (1/m)')
    plt.ylabel('Frequency (1/s)')
    cbar = plt.colorbar()
    cbar.set_label('log10(Amplitude)')
    if outfile:
        plt.savefig(outfile)
    plt.show()
    return