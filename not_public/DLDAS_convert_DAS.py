# Wir benötigen DAS zum Training im .mat Format
# open and run jupyter notebook:
# (DASDL) johanna@scadsnb239:~/JupyterProjects/DLDAS_Denoising-main$ jupyter notebook
from datetime import datetime, timedelta
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5


def inspect_mat73_file(file_path):
    """
    Inspect the structure of a MATLAB v7.3 .mat file and print detailed information about its contents.

    Parameters:
    file_path (str): Path to the .mat file

    Returns:
    h5py.File object
    """
    try:
        # Open the file in read mode
        f = h5py.File(file_path, 'r')

        def print_attrs(name, obj):
            """Helper function to print attributes of each object"""
            print("\n" + "-" * 50)
            print(f"Path: {name}")
            print(f"Type: {type(obj)}")

            if isinstance(obj, h5py.Dataset):
                print(f"Shape: {obj.shape}")
                print(f"Data type: {obj.dtype}")

                # Try to show a preview of the data
                try:
                    if obj.size < 10:
                        data = obj[()]
                        if isinstance(data, np.ndarray):
                            print("Preview:", data)
                        else:
                            print("Preview:", data)
                    else:
                        data = obj[0:min(5, obj.shape[0])]
                        print("First few elements:", data)
                except Exception as e:
                    print(f"Cannot preview data: {str(e)}")

            # Print attributes if any
            if len(obj.attrs) > 0:
                print("Attributes:")
                for key, value in obj.attrs.items():
                    print(f"  {key}: {value}")

        print(f"\nStructure of {file_path}:")
        print("=" * 50)

        # Visit each object in the file
        f.visititems(print_attrs)

        return f

    except Exception as e:
        print(f"Error reading .mat file: {str(e)}")
        return None


def get_dataset_value(f, path):
    """
    Helper function to get the value of a dataset, handling MATLAB's string encoding

    Parameters:
    f: h5py.File object
    path: str, path to the dataset

    Returns:
    The dataset value, with proper decoding for strings
    """
    try:
        dataset = f[path]
        if h5py.check_string_dtype(dataset.dtype):
            return ''.join(chr(c[0]) for c in dataset[()])
        return dataset[()]
    except Exception as e:
        print(f"Error getting dataset value: {str(e)}")
        return None

""" 
# Example usage
if __name__ == "__main__":
    # Replace with your .mat file path
    file_path = "/home/johanna/JupyterProjects/DLDAS_Denoising-main/data/eq-68.mat"
    f = inspect_mat73_file(file_path)

    if f is not None:
        # Keep the file open for further inspection if needed
        # When done, close the file
        f.close()
"""


file_path = "../data/raw_DAS/rhone1khz_UTC_20200727_050438.575.h5"

event_times = {0: ["2020-07-27 08:17:35.3", 1705, 2250, 1, "ALH"],
               5: ["2020-07-27 19:43:31.0", 1705, 2250, 1, "ALH"],

               20: ["2020-07-27 00:21:46.6", 1705, 2150, 2, "ALH"],

               82: ["2020-07-27 05:04:55.4", 1705, 2305, 3, "ALH"],
               }
seconds = 1.2
ids = [0, 5, 20, 82]

all_data = []

for id in ids:

    event_time = event_times[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_end = t_start + timedelta(seconds=1.2)
    ch_start = event_times[id][1]
    ch_end = event_times[id][2]
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir="../data/raw_DAS/", convert=False)
    data = data[:, ch_start:ch_end]

    print(headers)

    plt.figure(figsize=(15, 15))
    plt.imshow(data, vmax=10, vmin=-10)
    plt.show()

    #np.save("experiments/15_DASDL/training_data/rohnegltscher_sample_" + str(id) + ".npy", data)

    all_data.append(data)




combined_data = np.concatenate(all_data, axis = 1)
#np.save("experiments/15_DASDL/training_data/rohnegltscher_sample_combined.npy", combined_data)

plt.figure(figsize=(15,15))
plt.imshow(combined_data, vmax=10, vmin=-10)
plt.show()


