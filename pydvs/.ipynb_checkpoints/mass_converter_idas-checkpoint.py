import glob
import os
import pathlib
import sys
import warnings

import h5py
import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream as Stream
from tqdm import tqdm

from readers import convert_native_to_idas2
from readers import das_reader as reader
from readers import peak_h5_asdf_data, peak_h5_native_data, read_idas2_h5_file
from readers import stream_comparison as compare
from readers import write_idas2_data


def mass_converter_idas2(input_directory, output_directory, check_files_every_n=5, check_existing=True,
                         debug=False):
    """Mass converter, converting native silixa .h5 files from input_directory to converted idas .hdf5 files in output_directory

    Args:
        input_directory (pathlib.Path): path of input directory. Will browse all subdirectories for .h5 files
        output_directory (pathlib.Path): output directory. Function will create folders for each day separately
        check_files_every_n (int, optional): Frequency with which the input and output files are compared. If=1, then every file will compare its inout data to the data in the written file. This slows down everything significantly. Defaults to 5.
        check_existing (bool, optional): If a converted file exists based on the input file, then check if the file contents are the same. Note that this also only takes the check_files_every_n into account!. Defaults to True.
    """

    # Variable re-naming and setting up iterators
    in_folder = input_directory
    out_folder = output_directory
    it_converted = 0
    it_ignored = 0

    # All files that should be converted (make a list for the progressbar)
    files_to_convert = []
    for path, subdirs, files in os.walk(in_folder):
        for name in sorted(files):
            if name.endswith(".h5"):
                files_to_convert.append(pathlib.Path(path, name))

    # Make folder structures and establish the output structure
    for p in tqdm(files_to_convert, desc='Converting Files', ncols=75):
        tmp_file = "idas2_UTC_" + \
            p.name.split("_UTC_")[-1].split(".h5")[0] + \
            "_"+p.name.split("_UTC_")[0]+".hdf5"
        dayfolder = pathlib.Path(out_folder, p.name.split("_")[-2])
        dayfolder.mkdir(parents=True, exist_ok=True)

        target_folder = pathlib.Path(out_folder, p.name.split("_")[-2])
        target_folder.mkdir(parents=True, exist_ok=True)

        p_tmp_new_fname = pathlib.Path(dayfolder, tmp_file)

        # If the output file exists
        if (p_tmp_new_fname).is_file():
            if debug:
                warnings.warn(
                    "\n>>File Existing: \n\t"+str(p_tmp_new_fname.name), UserWarning)
            if check_existing:
                if ((it_ignored+it_converted) % int(check_files_every_n) == 0):
                    st_in = reader(p, h5type="native")
                    st_out = reader(p_tmp_new_fname, h5type="idas2")
                    compare(st_in, st_out, metadata=False, check_stats=True)
            it_ignored += 1
        # Create the output file
        else:
            if ((it_ignored+it_converted) % int(check_files_every_n) == 0):
                convert_native_to_idas2(
                    p, p_tmp_new_fname, debug=debug, test=True)
            else:
                convert_native_to_idas2(
                    p, p_tmp_new_fname, debug=debug, test=False)
            open(pathlib.Path(dayfolder, tmp_file), 'a').close()
            it_converted += 1
    # Finish up
    print("Files converted: \t", it_converted,
          "\nFiles already existing:\t", it_ignored)


if __name__ == "__main__":
    in_folder = "/Users/patrick/polybox/test_data/idas_readers/test_data/h5_native"
    tmp_folders = "/Users/patrick/polybox/test_data/idas_readers/test_data/h5_idas2/"

    mass_converter_idas2(in_folder, tmp_folders,
                         check_files_every_n=5, check_existing=True, debug=False)
