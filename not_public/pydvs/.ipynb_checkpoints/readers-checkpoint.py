import glob
import os
import pathlib
import warnings

import h5py
import numpy as np
import obspy
import pyasdf
from obspy import UTCDateTime
from obspy.core.stream import Stream as Stream


def get_data_attrib(ds):
    """
    takes an h5 dataset as input and returns the metadata
    """
    return ds.auxiliary_data.SystemInformation.sysinfo.parameters


def peak_h5_native_data(file):
    """
    Looks at a native silixa iDAS .h5 file and returns some fields from the metadata in the header
    """
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['Acoustic']
        dtype = dset.dtype

        ddims = dset.shape
        nsamp = ddims[0]
        nchan = ddims[1]

        metadata = dict(dset.attrs)
        starttime = UTCDateTime(str(metadata["ISO8601 Timestamp"]))

        fs = int(metadata["SamplingFrequency[Hz]"])
        dx = float(metadata["SpatialResolution[m]"])

        d0 = float(metadata["Start Distance (m)"])

        endtime = starttime + ((nsamp-1)/float(fs))

        return starttime, endtime, fs, dx, d0


def peak_h5_idas2_data(file):
    """
    Looks at a native silixa iDAS .h5 file and returns some fields from the metadata in the header
    """
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['raw_das_data']
        dtype = dset.dtype

        ddims = dset.shape
        nsamp = ddims[0]
        nchan = ddims[1]

        metadata = dict(dset.attrs)
        starttime = UTCDateTime(str(metadata["starttime"]))

        fs = int(metadata["sampling_frequency_Hz"])
        dx = float(metadata["spatial_resolution_m"])
        d0 = float(metadata["start_distance_m"])
        endtime = starttime + ((nsamp-1)/float(fs))

        return starttime, endtime, fs, dx, d0


def read_h5_native_file(file, stream=True, channels=[0, -1], auxiliary=True):
    """
    Read native silixa h5 file. 
    """
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['Acoustic']
        dtype = dset.dtype

        ddims = dset.shape
        nsamp = ddims[0]
        nchan = ddims[1]

        metadata = dict(dset.attrs)
        starttime = UTCDateTime(str(metadata["ISO8601 Timestamp"]))

        fs = int(metadata["SamplingFrequency[Hz]"])
        dx = float(metadata["SpatialResolution[m]"])

        d0 = float(metadata["Start Distance (m)"])

        # Data is in the format of (nsamp, nchan). So a single trace is dset[:,tracenumber]
        if channels == [0, -1]:
            data = np.array(dset[:, :])
        else:
            data = np.array(dset[:, channels[0]:channels[-1]])

    if stream:
        st = Stream()
        trace_l = (("sampling_rate", int(fs)),
                   ("delta", 1./int(fs)),
                   ("calib", 1.),
                   ("npts", int(nsamp)),
                   ("network", "XS"),
                   ("station", ""),
                   ("starttime", starttime))
        trace_dict = {key: value for (key, value) in trace_l}

        for i in range(data.shape[1]):
            tr = obspy.Trace(data=data[:, i], header=trace_dict)
            tr.stats.distance = d0 + (i+channels[0])*dx
            tr.stats.channel = "ESN"
            tr.stats.station = "D" + "{0:05d}".format(i+channels[0])
            st.__iadd__(tr)

        if auxiliary:
            st.metadata = metadata

        return(st)

    else:
        return(data, channels, metadata)


def read_h5_native_files(files, as_stream=True, stream=True, channels=[0, -1], auxiliary=True, merge=True, sort=True):
    """Reader for native silixa iDAS files. Subroutine of das_reader.

    Args:
        files ([type]): [description]
        as_stream (bool, optional): [description]. Defaults to True.
        stream (bool, optional): [description]. Defaults to True.
        channels (list, optional): [description]. Defaults to [0, -1].
        auxiliary (bool, optional): [description]. Defaults to True.
        merge (bool, optional): [description]. Defaults to True.
        sort (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if not isinstance(files, list):
        files = [files]

    # CHECKING ACQUISITION PARAMETERS AND GAPS IN THE DATA BEFORE READING IN
    for i in range(len(files)-1):
        starttime0, endtime0, fs0, dx0, d00 = peak_h5_native_data(files[i])
        starttime1, endtime1, fs1, dx1, d01 = peak_h5_native_data(files[i+1])

        if (fs0 == fs1) & (dx0 == dx1) & (d00 == d01):
            pass
        else:
            warnings.warn("Different acquisition parameters. \n>> Falling back to as_stream=True and stream=True with merge=False",
                          UserWarning)
            as_stream = True
            stream = True
            merge = False
            break

        if starttime1 == endtime0+(1./fs1):
            pass
            rerun = False
        else:
            warnings.warn("Misaligned data (gaps or overlap).\n>> Falling back to as_stream=True and stream=True with merge=False",
                          UserWarning)
            as_stream = True
            stream = True
            merge = False
            break

    if as_stream:
        stream = True
        # WORKING WITH STREAM OBJECTS DIRECTLY (one stream per file)
        st = Stream()
        iter_ = (i for i in range(len(files)))
        for file in files:
            i = next(iter_)
            st_tmp = read_h5_native_file(
                file, stream=stream, channels=channels, auxiliary=auxiliary)
            st += st_tmp
            if i == 0:
                st.metadata = st_tmp.metadata

        if merge:
            if not st.get_gaps():
                st.merge(method=1).sort()
            else:
                warnings.warn("Gaps or overlap in the data. Returned stream object is not merged!",
                              UserWarning)
                if True:
                    st.print_gaps()
        if sort:
            st.sort()
        return st

    else:
        iter_ = (i for i in range(len(files)))
        for file in files:
            i = next(iter_)
            if i == 0:
                data, channels, metadata = read_h5_native_file(
                    file, stream=False, channels=channels, auxiliary=auxiliary)
            else:
                data_tmp, _, _ = read_h5_native_file(
                    file, stream=False, channels=channels, auxiliary=auxiliary)
                data = np.vstack([data, data_tmp])

        if stream:
            if not merge:
                warnings.warn(
                    'merge=True was set because of working with the numpy arrays', UserWarning)
            fs = int(metadata["SamplingFrequency[Hz]"])
            dx = float(metadata["SpatialResolution[m]"])
            d0 = float(metadata["Start Distance (m)"])
            starttime = UTCDateTime(str(metadata["ISO8601 Timestamp"]))
            st = Stream()
            trace_l = (("sampling_rate", int(fs)),
                       ("delta", 1./int(fs)),
                       ("calib", 1.),
                       ("npts", int(data.shape[0])),
                       ("network", "XS"),
                       ("station", ""),
                       ("starttime", starttime))
            trace_dict = {key: value for (key, value) in trace_l}

            for i in range(data.shape[1]):
                tr = obspy.Trace(data=data[:, i], header=trace_dict)
                tr.stats.distance = d0 + (i+channels[0])*dx
                tr.stats.channel = "ESN"
                tr.stats.station = "D" + "{0:05d}".format(i+channels[0])
                st.__iadd__(tr)

            if auxiliary:
                st.metadata = metadata

            if sort:
                st.sort()
            return(st)

        else:
            return(data, channels, metadata)


def peak_h5_asdf_data(file):
    """Peak asdf file to get some overview on the data

    Args:
        file pathlib.Path : File to peak

    Returns:
        list: starttime, endtime, sampling frequency, spatial resolution and start distance of array
    """
    file = pathlib.Path(file)
    with pyasdf.ASDFDataSet(file, mode="r") as ds:
        metadata = get_data_attrib(ds)

        fs = int(metadata["SamplingFrequency[Hz]"])
        dx = float(metadata["SpatialResolution[m]"])
        d0 = float(metadata["Start Distance (m)"])

        tr = ds.waveforms["XS.D00000"].raw_data[0]
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime

        return starttime, endtime, fs, dx, d0


def read_h5_asdf_file(file, stream=True, channels=[0, -1], auxiliary=True):
    """Subroutine called by read_h5_asdf_files. 

    Args:
        file ([type]): [description]
        stream (bool, optional): [description]. Defaults to True.
        channels (list, optional): [description]. Defaults to [0, -1].
        auxiliary (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    st = Stream()
    with pyasdf.ASDFDataSet(file, mode="r") as ds:

        if channels == [0, -1]:
            wlist = ds.waveforms.list()
        else:
            wlist = ds.waveforms.list()[channels[0]:channels[1]]
        for tr in wlist:
            st.__iadd__(ds.waveforms[tr].raw_data)

        if auxiliary:
            st.metadata = get_data_attrib(ds)
            d0 = float(st.metadata["Start Distance (m)"])
            dx = float(st.metadata["SpatialResolution[m]"])
            iter_ = (i for i in range(len(st)))

            for tr in st:
                i = next(iter_)
                tr.stats.distance = d0 + i*dx
    return st


def read_h5_asdf_files(files, stream=True, channels=[0, -1], auxiliary=True, merge=True, sort=True):
    """Subroutine to read in asdf files. Called by das_reader, so please look at that function for more info

    Args:
        files ([type]): [description]
        stream (bool, optional): [description]. Defaults to True.
        channels (list, optional): [description]. Defaults to [0, -1].
        auxiliary (bool, optional): [description]. Defaults to True.
        merge (bool, optional): [description]. Defaults to True.
        sort (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    if not isinstance(files, list):
        files = [files]

    for i in range(len(files)-1):
        starttime0, endtime0, fs0, dx0, d00 = peak_h5_asdf_data(files[i])
        starttime1, endtime1, fs1, dx1, d01 = peak_h5_asdf_data(files[i+1])

        if (fs0 == fs1) & (dx0 == dx1) & (d00 == d01):
            pass
        else:
            warnings.warn("Different acquisition parameters. \n>> falling back to merge=False",
                          UserWarning)
            merge = False

        if starttime1 == endtime0+(1./fs1):
            pass
        else:
            warnings.warn("Misaligned data (gaps or overlap).\n>> Falling back to merge=False",
                          UserWarning)
            merge = False

    # WORKING WITH STREAM OBJECTS DIRECTLY (one stream per file)
    st = Stream()
    iter_ = (i for i in range(len(files)))
    for file in files:
        i = next(iter_)
        st_tmp = read_h5_asdf_file(
            file, stream=stream, channels=channels, auxiliary=auxiliary)
        st += st_tmp
        if i == 0:
            if auxiliary:
                st.metadata = st_tmp.metadata

    if merge:
        if not st.get_gaps():
            st.merge(method=1).sort()
        else:
            warnings.warn("Gaps or overlap in the data. Returned stream object is not merged!",
                          UserWarning)
            if True:
                st.print_gaps()
    if sort:
        st.sort()
    return st


def read_idas2_h5_file(file, stream=True, channels=[0, -1], auxiliary=True):
    file = pathlib.Path(file)
    with h5py.File(file, "r") as f:
        dset = f['raw_das_data']
        dtype = dset.dtype

        ddims = dset.shape
        nsamp = ddims[0]
        nchan = ddims[1]

        metadata = dict(dset.attrs)
        starttime = UTCDateTime(str(metadata["starttime"]))

        fs = int(metadata["sampling_frequency_Hz"])
        dx = float(metadata["spatial_resolution_m"])
        d0 = float(metadata["start_distance_m"])

        if channels == [0, -1]:
            data = np.array(dset[:, :])
        else:
            data = np.array(dset[:, channels[0]:channels[-1]])

        if stream:
            st = Stream()
            trace_l = (("sampling_rate", int(fs)),
                       ("delta", 1./int(fs)),
                       ("calib", 1.),
                       ("npts", int(nsamp)),
                       ("network", "XS"),
                       ("station", ""),
                       ("starttime", starttime))
            trace_dict = {key: value for (key, value) in trace_l}

            for i in range(data.shape[1]):
                tr = obspy.Trace(data=data[:, i], header=trace_dict)
                tr.stats.distance = d0 + (i+channels[0])*dx
                tr.stats.channel = "ESN"
                tr.stats.station = "D" + "{0:05d}".format(i+channels[0])
                st.__iadd__(tr)

            if auxiliary:
                st.metadata = metadata

            return(st)
        else:
            return(data, channels, metadata)


def read_idas2_h5_files(files, as_stream=False, stream=True, channels=[0, -1], auxiliary=True, merge=True, sort=True):
    """Reader function for idas2 .hdf5 files. Subroutine called by das_reader, so for details, please look at das_reader.

    Args:
        files ([type]): [description]
        as_stream (bool, optional): [description]. Defaults to False.
        stream (bool, optional): [description]. Defaults to True.
        channels (list, optional): [description]. Defaults to [0, -1].
        auxiliary (bool, optional): [description]. Defaults to True.
        merge (bool, optional): [description]. Defaults to True.
        sort (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if not isinstance(files, list):
        files = [files]

    for i in range(len(files)-1):
        starttime0, endtime0, fs0, dx0, d00 = peak_h5_idas2_data(files[i])
        starttime1, endtime1, fs1, dx1, d01 = peak_h5_idas2_data(files[i+1])

        if (fs0 == fs1) & (dx0 == dx1) & (d00 == d01):
            pass
        else:
            warnings.warn("Different acquisition parameters. \n>> falling back to merge=False",
                          UserWarning)
            as_stream = True
            stream = True
            merge = False
            break

        if starttime1 == endtime0+(1./fs1):
            pass
        else:
            warnings.warn("Misaligned data (gaps or overlap).\n>> Falling back to merge=False",
                          UserWarning)
            as_stream = True
            stream = True
            merge = False
            break

    if as_stream:
        stream = True
        # WORKING WITH STREAM OBJECTS DIRECTLY (one stream per file)
        st = Stream()
        iter_ = (i for i in range(len(files)))
        for file in files:
            i = next(iter_)
            st_tmp = read_idas2_h5_file(
                file, stream=stream, channels=channels, auxiliary=auxiliary)
            st += st_tmp
            if i == 0:
                st.metadata = st_tmp.metadata

        if merge:
            if not st.get_gaps():
                st.merge(method=1).sort()
            else:
                warnings.warn("Gaps or overlap in the data. Returned stream object is not merged!",
                              UserWarning)
                if True:
                    st.print_gaps()
        if sort:
            st.sort()
        return st

    else:
        iter_ = (i for i in range(len(files)))
        for file in files:
            i = next(iter_)
            if i == 0:
                data, channels, metadata = read_idas2_h5_file(
                    file, stream=False, channels=channels, auxiliary=auxiliary)
            else:
                data_tmp, _, _ = read_idas2_h5_file(
                    file, stream=False, channels=channels, auxiliary=auxiliary)
                data = np.vstack([data, data_tmp])

        if stream:
            if not merge:
                warnings.warn(
                    'merge=True was set because of working with the numpy arrays', UserWarning)

            starttime = UTCDateTime(str(metadata["starttime"]))
            fs = int(metadata["sampling_frequency_Hz"])
            dx = float(metadata["spatial_resolution_m"])
            d0 = float(metadata["start_distance_m"])
            # endtime = starttime + ((nsamp-1)/float(fs))

            st = Stream()
            trace_l = (("sampling_rate", int(fs)),
                       ("delta", 1./int(fs)),
                       ("calib", 1.),
                       ("npts", int(data.shape[0])),
                       ("network", "XS"),
                       ("station", ""),
                       ("starttime", starttime))
            trace_dict = {key: value for (key, value) in trace_l}

            for i in range(data.shape[1]):
                tr = obspy.Trace(data=data[:, i], header=trace_dict)
                tr.stats.distance = d0 + (i+channels[0])*dx
                tr.stats.channel = "ESN"
                tr.stats.station = "D" + "{0:05d}".format(i+channels[0])
                st.__iadd__(tr)

            if auxiliary:
                st.metadata = metadata

            if sort:
                st.sort()
            return(st)

        else:
            return(data, channels, metadata)


def das_reader(files, auxiliary=True, sort=True, merge=True,
               stream=True, as_stream=False,
               channels=[0, -1], h5type='native', debug=False):
    """
    Reader function to read in iDAS data into either a numpy array or an obspy stream object. Default values should be: auxiliary=True, as_stream=False, and stream=True. 
    :type files: list
    :param files: List of files to read
    :type auxiliary: bool
    :param auxiliary: If metadata (header) should be read in and saved to st.metadata
    :type sort: bool
    :param sort: If stream should be sorted by station
    :type merge: bool
    :param merge: If the stream object should be merged or not before returning
    :type strean: bool
    :param stream: If True, the function will return an obspy Stream object. If False, will return data, channels, metadata, where data is the read in data as 2d numpy array
    :type as_stream: bool
    :param as_stream: Decide how to read in and handle the data inside the reader. default to False, but if h5type is asdf, this will automatically be set to True. Same if there are gaps in the data, it will fall back to as_stream=True, because the matrix/numpy version can not handle gaps due to lacking of timestamping in numpy arrays
    :type channels: list of 2 values
    :param channels: Slice of channels to be read in. If default [0,-1], the entire data will be read in. Otherwise arbitrary slicing would be possible (eg. [20,50]). No interval slicing implemented yet (like [20:50:10] ). This is on the ToDo.
    :type h5type: str
    :param h5type: The type of the data. Either native silixa h5 as from the iDAS (h5type='native') or converted asdf data (h5type='asdf'). Now also works for idas2 .hdf5 files: (h5type='ida2')
    :type debug: bool
    :param debug: optional print outputs
    """
    if not isinstance(files, list):
        files = [files]

    if h5type not in ['asdf', 'native', 'idas2']:
        sys.exit('Invalid h5type specified')
    if debug:
        print('\U0001F50D Reading in: \n', files)
    if h5type == 'native':
        if stream:
            st = read_h5_native_files(files, auxiliary=auxiliary, sort=sort, merge=merge,
                                      stream=stream, as_stream=as_stream,
                                      channels=channels)
            if debug:
                print("\U00002714 success")
            return st
        else:
            data, channels, metadata = read_h5_native_files(files, auxiliary=auxiliary, sort=sort, merge=merge,
                                                            stream=stream, as_stream=as_stream,
                                                            channels=channels)
            if debug:
                print("\U00002714 success")
            return (data, channels, metadata)

    if h5type == 'idas2':
        if stream:
            st = read_idas2_h5_files(files, auxiliary=auxiliary, sort=sort, merge=merge,
                                     stream=stream, as_stream=as_stream,
                                     channels=channels)
            if debug:
                print("\U00002714 success")
            return st
        else:
            data, channels, metadata = read_idas2_h5_files(files, auxiliary=auxiliary, sort=sort, merge=merge,
                                                           stream=stream, as_stream=as_stream,
                                                           channels=channels)
            if debug:
                print("\U00002714 success")
            return (data, channels, metadata)

    if h5type == 'asdf':
        st = read_h5_asdf_files(files, stream=stream, channels=channels,
                                auxiliary=auxiliary, merge=merge, sort=sort)
        if debug:
            print("\U00002714 success")
        return st


def stream_comparison(st0, st1, metadata=False, check_stats=False):
    """ 
    Compare two streams to each other, based on start/endtimes and data content in each trace. Also compares the data_attrib 

    :type st0: obspy.core.stream
    :param st0: first stream to compare with second stream

    :type st1: obspy.core.stream
    :param st1: second stream to compare with first stream

    return: boolean if streams are the same or not. 
    Both streams need an .metadata attribute!

    """
    # try:
    if metadata:
        if st0.metadata == st1.metadata:
            pass
        else:
            print('Data Attributes different!')
            return False

    if len(st0) == len(st1):
        pass
    else:
        print("Lengths not the same")
        return False

    if st0[0].stats.starttime == st1[0].stats.starttime:
        if st0[0].stats.endtime == st1[0].stats.endtime:
            pass
        else:
            print("end times dont align")
            return False
    else:
        print("Start times do not align")
        return False

    for i in range(len(st0)):
        try:
            np.testing.assert_array_equal(
                st0[i].data, st1[i].data, "Data content of sorted array is not equal")
        except AssertionError as err:
            print(err)
            return False
        if check_stats:
            if not st0[i].stats == st1[i].stats:
                warnings.warn("\n>>Trace stats are differen!", UserWarning)
                print("stats")
                return False
        else:
            pass

    return True


def trace_comparison(tr1, tr2):
    """Compare two obspy Traces with each other

    Args:
        tr1 ([type]): Trace 1
        tr2 ([type]): Trace 2

    Returns:
        bool: Are both traces the same ? 
    """
    if tr1.stats.station != tr2.stats.station:
        print('TRACE stations not matching',
              tr1.stats.station, tr2.stats.station)
        return False

    if tr1.stats.starttime == tr2.stats.starttime:
        if tr1.stats.endtime == tr2.stats.endtime:
            pass
        else:
            print("end times dont align")
            return False

    try:
        np.testing.assert_array_equal(
            tr1.data, tr2.data, "Data content of sorted array is not equal")
    except AssertionError as err:
        print(err)
        return False
    return True


def stream_to_data(st):
    """ Convert stream to 2D numpy array 

    Args:
        st (obspy Stream): Input stream

    Returns:
        2D numpy array : Output array
    """
    arr = []
    for tr in st:
        arr.append(np.array(tr.data))
    arr = np.array(arr)
    return arr


def data_to_stream(arr, st):
    """ Inverse of stream_to_data

    Args:
        arr (2D numpy array): Array to convert to stream
        st (obspy Stream): return data into this stream

    Returns:
        obspy stream: obspy stream based on the data from arr
    """
    stream = st.copy()
    i = 0
    if len(arr) == len(st):
        for tr in stream:
            tr.data = arr[i]
            i += 1
        return stream
    else:
        warnings.warn('ERROR in the data_to_stream conversion!')
        return None


def write_idas2_data(file_out, data, metadata):
    """Write idas2 hdf5 data based on the file_out path, data and metadata

    Args:
        file_out ([type]): pathlib.Path filename of the output file (complete path)
        data ([type]): 2D array containing the data
        metadata ([type]): dictionary with metadata
    """
    with h5py.File(file_out, "w") as f:
        dset = f.create_dataset("raw_das_data", data=data, dtype=data.dtype)
        for k in metadata:
            dset.attrs[k] = str(metadata[k])


def convert_native_to_idas2(file_in, file_out, debug=False, test=True):
    """Converts input native silixa .h5 file to idas2 .hdf file

    Args:
        file_in ([type]): pathlib.Path filename for input file
        file_out ([type]): pathlib.Path filename for new file
        debug (bool, optional): Show debug information. Defaults to False.
        test (bool, optional): Test file contents (data and attributes) between input and output files. Defaults to True.
    """

    starttime, endtime, fs, dx, d0 = peak_h5_native_data(file_in)
    data, channels, metadata = das_reader(
        file_in, channels=[0, -1], h5type='native', debug=debug, stream=False)

    new_metadata = {}

    metakeys = ['SamplingFrequency[Hz]',
                'SpatialResolution[m]',
                'GaugeLength',
                'Acoustic Output',
                'GPSTimeStamp',
                'CPUTimeStamp',
                'ISO8601 Timestamp',
                'MeasureLength[m]',
                'OutputDecimation[bool]',
                'P',
                'PreTrigSamples',
                'Precise Sampling Frequency (Hz)',
                'PulseWidth[ns]',
                'Start Distance (m)',
                'StartPosition[m]',
                'Stop Distance (m)',
                'Fibre Length Multiplier',
                'FibreIndex',
                'SystemInfomation.GPS.Altitude',
                'SystemInfomation.GPS.Latitude',
                'SystemInfomation.GPS.Longitude',
                'SystemInfomation.GPS.SatellitesAvailable',
                'SystemInfomation.GPS.UTCOffset',
                'SystemInfomation.OS.HostName',
                'Time Decimation',
                'Zero Offset (m)',
                'iDASVersion',
                'SavingBandwidth (MB/s)',
                'SystemInfomation.ProcessingUnit.FPGA1.TempReadings']

    new_meta_keys = ["sampling_frequency_Hz",
                     "spatial_resolution_m",
                     "gauge_length_m",
                     "acoustic_output__",
                     "gps_time_stamp",
                     "cpu_time_stamp",
                     "iso_8601_time_stamp",
                     "measure_length_m",
                     "output_decimation",
                     "p_value",
                     "pre_trigger_samples",
                     "precise_sampling_frequency_Hz",
                     "pulse_width_ns",
                     "start_distance_m",
                     "start_position_m",
                     "stop_distance_m",
                     "fibre_length_multiplier",
                     "fibre_index",
                     "gps_altitude",
                     "gps_latitude",
                     "gps_longitude",
                     "gps_number_of_satellites",
                     "gps_utc_offset",
                     "host_name",
                     "time_decimation",
                     "zero_offset_m",
                     "das_version",
                     "saving_bandwidth_mb/s",
                     "temperature_readings"]

    new_metadata['starttime'] = str(starttime)
    new_metadata['endtime'] = str(endtime)
    new_metadata['samples'] = str(data.shape[0])
    new_metadata['channels'] = str(data.shape[1])

    for i in range(len(metakeys)):
        new_metadata[new_meta_keys[i]] = metadata[metakeys[i]]

    write_idas2_data(file_out, data, new_metadata)

    if test:
        st_in = das_reader(file_in, h5type="native")
        st_out = das_reader(file_out, h5type="idas2")
        stream_comparison(st_in, st_out, metadata=False, check_stats=True)
        stream_comparison(st_in, st_out, metadata=False, check_stats=True)

        # data_new, _, _ = das_reader(file_out, h5type='idas2', stream=False)
        # try:
        #     np.testing.assert_array_equal(
        #         data, data_new, "Data content of sorted array is not equal")
        # except AssertionError as err:
        #     print(err)
        #     return False

        # STREAM TEST
        # st_in  = das_reader(file_in, h5type="native")
        # st_out = das_reader(file_out, h5type="idas2")
        # stream_comparison(st_in, st_out, metadata=False, check_stats=True)
