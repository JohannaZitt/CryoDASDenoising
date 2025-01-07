
import os

# Generate .txt file for Producing cwt scale data



data_folder_path = "/home/johanna/JupyterProjects/DLDAS_Denoising_Omar_Saad/numpy_data"
data_files = os.listdir(data_folder_path)

with open("experiments/15_DASDL/terminal_commands.txt", "a") as file:

    for data_file in data_files:
        print_text = "matlab -nodisplay -batch \"A_computing_cwt('numpy_data/" + data_file + "', 'm_data/cwt_" + data_file[:-4] + ".mat')\""
        file.write(print_text + "\n")


