


# Introduction 


This software repository contains the scripts necessary to reproduce the results from the article "Self-Supervised 
Coherence-Based Denoising on Cryoseismological Distributed Acoustic Sensing Data." The preprint of the article is available 
on authorea: [DOI: 10.22541/au.172779667.76811452/v2](https://doi.org/10.22541/au.172779667.76811452/v2)


### Abstract:

One major challenge in cryoseismology is that signals of interest are often buried within
the high noise level emitted by a multitude of environmental processes. Events of interest
potentially stay unnoticed and remain unanalyzed, particularly because conventional
sensors cannot monitor an entire glacier. However, with Distributed Acoustic Sensing
(DAS), we can observe seismicity over multiple kilometers. DAS systems turn common
fiber-optic cables into seismic arrays that measure strain rate data, enabling researchers
to acquire seismic data in hard-to-access areas with high spatial and temporal resolution.
We deployed a DAS system on Rhonegletscher, Switzerland, using a 9 km long fiber-optic 
cable that covered the entire glacier, from its accumulation to its ablation zone,
recording seismicity for one month. The highly active and dynamic cryospheric environment, 
in combination with poor coupling, resulted in DAS data characterized by a low
Signal-to-Noise Ratio (SNR) compared to classical point sensors. Our objective is to effectively 
denoise this dataset. We use a self-supervised J -invariant U-net autoencoder capable of separating incoherent 
environmental noise from temporally and spatially coherent signals of interest (e.g.,
stick-slip or crevasse signals). The method shows enhanced inter-channel coherence, increased
SNR, and significantly improved visibility of the icequakes. Further, we compare
different training data types varying in recording position, wavefield component, and waveform 
diversity. Our approach has the potential to enhance the detection capabilities of
events of interest in cryoseismological DAS data, hence to improve the understanding
of processes within Alpine glaciers.




# Setup

1. Download the code using the terminal

       gh repo clone JohannaZitt/MAIN_DAS_denoising

    Alternatively, download the code via the GitHub user interface:

       Code -> Download ZIP

    and decompress the folder.


2. Install and activate the environment environment.yml via conda:

       conda env create -f environment.yml
       conda activate env_main_das_denoising

3. Download the data folder [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13868934.svg)](https://doi.org/10.5281/zenodo.13868934), 
   unzip it, and replace the empty data folder in the Jupyter project with the downloaded one.

**Optional:**
4. To simplify code execution without the need to denoise the real-world data, the denoised data from experiment 
   03_accumulation_horizontal is provided: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13890738.svg)](https://doi.org/10.5281/zenodo.13890738).
   Download the folder, unzip it and place it under the directory `experiments/03_accumulation_horizontal`.




# Description of single .py files

### calculating_cc.py

Calculates the local waveform coherence and cross-correlation between DAS data and co-located seismometer data for each 
experiment, as detailed in `Section 4.1.1` and `Section 4.3`. The results are saved in a .csv file named `cc_evaluation_id.csv` 
within the respective experiment folder.

**Requirements:** To execute this script for all experiments, the raw DAS data must be denoised beforehand. 
For experiment 03_accumulation_horizontal, you can download the denoised data (Optional Step 4. in Setup).


### denoise_synthetic_DAS.py

Denoises the synthetic data located in the folders `data/synthetic_DAS/from_DAS` and `data/synthetic_DAS/from_seis`, 
as detailed in `Section 3.4`. The denoised data is saved in the directory: `experiments/experiment_name/denoised_synthetic_DAS`.


### denoising_DAS.py

Denoises the raw real-world data located in the folder `data/raw_DAS`, as outlined in `Section 3.4`. 
The denoised data is stored in the directory `experiments/experiment_name/denoisedDAS`.


### generate_DAS_preprocessed_training_data.py

The initial waveforms are generated from real-world DAS data sections for model fine-tuning as detailed in `Section 3.3`.


### generate_seismometer_preprocessed_training_data.py

The initial waveforms are generated from seismometer data for model training as detailed in `Section 3.2`.


### generate_synthetic_test_data.py

Generates synthetic test data from seismometer data and sections of DAS data. This generated test data is utilized in `Figure 3` and `Figure S2`.


### helper_functions.py

Includes several functions that are used repeatedly: e.g., bandpass filter and computation of local waveform coherence.


### main_training.py

Primary training is conducted on the initial waveforms generated using seismometer data.


### models.py

Class for the model structure and training data generator.


### plotting_fig3.py

Generates `Figure 3`: Raw data sections, denoised data sections and wiggle comparison for synthetically generated DAS data 
derived from seismometer data.

**Requirements:** Denoise synthetically generated DAS data obtained from seismometer data with denoiser 
`03_accumulation_horizontal` or download denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig4.py

Generates `Figure 4`: Raw and denoised real-world DAS data sections with local waveform coherence and wiggle comparison.

**Requirements:** Denoise real-world DAS data obtained with denoiser `03_accumulation_horizontal` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig5.py

Generates `Figure 5`: Frequency content of denoised and raw DAS records

**Requirements:**  Denoise real-world DAS data obtained with denoiser `03_accumulation_horizontal` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig6.py

Generates `Figure 6`: Average local waveform coherence and mean cross-correlation between DAS data and co-located seismometer data. 
The calculated cc values are saved in `experiments/experiment_name/cc_evaluation.csv` for each experiment under its respective name.


### plotting_figS1.py

Generates `Figure S1`: Representative initial waveforms.


### plotting_figS2.py

Generates `Figure S2`: Raw data sections, denoised data sections and wiggle comparison for synthetically generated DAS data 
derived from real-world DAS data.

**Requirements:** Denoise synthetically generated DAS data obtained from DAS data with denoiser 
`03_accumulation_horizontal` or download denoised DAS data sections (Optional Step 4. in Setup).


### plotting_figS3-S5.py

Generates `Figure S3`, `Figure S4`, and `Figure S5`: Waveform section plots for three exemplary DAS data sections displaying icequakes.

**Requirements:** Denoise real-world DAS data obtained with denoiser `03_accumulation_horizontal` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### plotting_fig6.py

Generates `Figure S6`: STA/LTA trigger of raw and denoised DAS data section. 

**Requirements:** Denoise real-world DAS data obtained with denoiser `03_accumulation_horizontal` or download the 
denoised DAS data sections (Optional Step 4. in Setup).


### retraining.py

Fine-tuning the models with real-world DAS data sections as described in `Section 3.3`.




# Help
If you need assistance, have any questions, or have suggestions for improvements, feel free to contact me via [email](mailto:johanna.zitt@uni-leipzig.de).



# References
Parts of the code are built upon the software provided by van den Endet et al. [1].

[1] van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A. ve Richard, C. (2021, 3 Mart). A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data. figshare. doi:10.6084/m9.figshare.14152277.v1