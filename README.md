


# Introduction 


This software repository contains the scripts necessary to reproduce the results from the article "Self-Supervised 
Coherence-Based Denoising of Cryoseismological Distributed Acoustic Sensing Data." The preprint of the article is available 
on authorea: [DOI: 10.22541/au.172779667.76811452/v2](https://doi.org/10.22541/au.172779667.76811452/v2)


### Abstract:

A major challenge in cryoseismology is that signals of interest are often buried within the high noise level emitted by a variety of environmental processes. Particular Distributed Acoustic Sensing (DAS) data often suffers from low signal-to-noise ratios (SNR) potentially resulting in a multitude of undetected events of interest, which further remain unanalyzed. To record seismicity, we deployed a DAS system on Rhône Glacier, Switzerland, using a 9 km long fiber-optic cable that covered the entire glacier, from its accumulation to its ablation zone. The highly active and dynamic cryospheric environment, in combination with poor coupling, resulted in DAS data characterized by a low SNR. Our objective is to develop and evaluate a method to effectively denoise this cryoseismological DAS dataset, while comparing our approach to state-of-the-art filtering and denoising methods. We propose the J-invariant-cryo denoiser, specifically trained on cryoseismological data and capable of separating incoherent environmental noise from temporally and spatially coherent signals of interest, based on a self-supervised J-invariant U-Net autoencoder. The method enhances inter-channel coherence, improves waveform similarity with co-located seismometers, and increases SNR. The comparison of different methods shows that our approach obtains the highest gain in SNR and highest similarity with co-located seismometers, while suffering from denoising artifacts in rare cases. The proposed denoiser has the potential to enhance the detection capabilities of events of interest in cryoseismological DAS data, hence to improve the understanding of processes within Alpine glaciers. 




# Setup

1. Download the code using the terminal

       gh repo clone JohannaZitt/CryoDASDenoising
    Alternatively, download the code via the GitHub user interface:

       Code -> Download ZIP

    and decompress the folder.


2. Install and activate the environment environment.yml via conda:

       conda env create -f environment.yml
       conda activate env_main_das_denoising

3. Download the data folder [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14998653.svg)](https://doi.org/10.5281/zenodo.14998653), 
   unzip it, and replace the empty data folder in the project with the downloaded one.

**Optional:**
4. To simplify code execution without the need to denoise the real-world data, the denoised data from experiment 
   02_accumulation is provided: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14999918.svg)](https://doi.org/10.5281/zenodo.14999918)
   Download the folder, unzip it and place it under the directory `experiments/02_accumulation`.




# Description of single .py files

### calculating_cc.py

Calculates the local waveform coherence and cross-correlation between DAS data and co-located seismometer data, and snr for each 
experiment, as detailed in `Section 4.3.2`. The results are saved in a .csv file named `cc_evaluation_id.csv` 
within the respective experiment folder.

**Requirements:** To execute this script for all experiments, the raw DAS data must be denoised beforehand. 
For experiment 02_accumulation, you can download the denoised data (Optional Step 4. in Setup).


### cc_values

Calculates the values for Table 1 in the paper. 


### denoise_synthetic_DAS.py

Denoises the synthetic data located in the folders `data/synthetic_DAS/from_seis`, 
as detailed in `Section 3.3`. The denoised data is saved in the directory: `experiments/experiment_name/denoised_synthetic_DAS`.


### denoising_DAS.py

Denoises the raw field data located in the folder `data/raw_DAS`, as outlined in `Section 3.3`. 
The denoised data is stored in the directory `experiments/experiment_name/denoisedDAS`.


### denoising_conventional.py
Denoises the raw field data located in the folder `data/raw_DAS`, as outlined in `Section 4.3`. 


### generate_DAS_preprocessed_training_data.py

The initial waveforms are generated from field DAS data sections for fine-tuning of the model J-invariant-earth-cryo.


### generate_seismometer_preprocessed_training_data.py

The initial waveforms are generated from seismometer data for model training as detailed in `Section 3.2`.


### generate_synthetic_test_data.py

Generates synthetic test data from seismometer data and sections of DAS data. This generated test data is utilized in `Figure 2` and `Figure S2`.


### helper_functions.py

Includes several functions that are used repeatedly: e.g., bandpass filter and computation of local waveform coherence.


### main_training.py

Primary training is conducted on the initial waveforms generated using seismometer data.


### models.py

Class for the model structure and training data generator.

### plotting_fig2.py

Generates `Figure 2`: Synthetic DAS data sections, denoised DAS data sections and time series comparison for synthetically generated DAS data derived from seismometer data.

**Requirements:** Denoise synthetically generated DAS data obtained from seismometer data with denoiser 
`02_accumulation` or download denoised DAS data sections (Optional Step 4. in Setup).

### plotting_fig3.py

Generates `Figure 3`: Raw and denoised field DAS sections with local waveform coherence and time series comparison.

**Requirements:** Denoise real-world DAS data obtained with denoiser `02_accumulation` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig4.py

Generates `Figure 4`: Frequency content of denoised and raw DAS records

**Requirements:**  Denoised field DAS data obtained with denoiser `02_accumulation` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig5_S7_S8.py

Generates `Figure 5`, `Figure S7`, and `Figure S8`: Performance comparison of different denoising models

**Requirements:**  Raw DAS data (02_accumulation_denoisedDAS) and denoised data for all models (experiment_denoisedDAS_image, Optional Step 4. in Setup).


### plotting_figS2.py


Generates `Figure S2`: Raw data sections, denoised data sections and wiggle comparison for synthetically generated DAS data derived from field DAS data.

**Requirements:** Denoise synthetically generated DAS data obtained from DAS data with denoiser 
`02_accumulation` or download denoised DAS data sections (Optional Step 4. in Setup).


### plotting_figS3-S5.py

Generates `Figure S3`, `Figure S4`, and `Figure S5`: Waveform section plots for three exemplary DAS data sections displaying icequakes.

**Requirements:** Denoise real-world DAS data obtained with denoiser `02_accumulation` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_figS6.py

Generates `Figure S6`: STA/LTA trigger of raw and denoised DAS data section. 

**Requirements:** Denoise field DAS data obtained with denoiser `02_accumulation` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_supplements.py

Generates all plots presented in plots/test_data.

**Requirements:** Denoise field DAS data obtained with denoiser `02_accumulation` or download the 
denoised DAS data sections (Optional Step 4. in Setup).




# Help
If you need assistance, have any questions, or have suggestions for improvements, feel free to contact me via [email](mailto:johanna.zitt@uni-leipzig.de).



# References
Parts of the code are built upon the software provided by van den Endet et al. [1].

[1] van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A. ve Richard, C. (2021, 3 Mart). A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data. figshare. doi:10.6084/m9.figshare.14152277.v1
