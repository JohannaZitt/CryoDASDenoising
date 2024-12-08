
import csv
import os

import matplotlib.pyplot as plt
import numpy as np




#experiments = ["01_ablation_horizontal", "03_accumulation_horizontal", "07_retrained_combined200", "11_vanende", "12_vanende_finetuned_cryo",
#               "13_isken_filter", "14_julius_filter"]
experiments = ["15_DASDL"]




for experiment in experiments:

    """ Path to cc values """
    csv_path = os.path.join("experiments", experiment, "cc_evaluation_" + experiment[0:2] + ".csv")

    # empty lists for value storage
    ablation_cc_gain = []
    ablation_cc_gain_seis = []
    ablation_snr_power = []
    ablation_snr_variance = []
    ablation_snr_rms = []
    ablation_snr_absolute = []

    accumulation_cc_gain = []
    accumulation_cc_gain_seis = []
    accumulation_snr_power = []
    accumulation_snr_variance = []
    accumulation_snr_rms = []
    accumulation_snr_absolute = []

    with open(csv_path, "r") as file:

        """ Read cc values """
        csv_reader = csv.DictReader(file)

        for row in csv_reader:


            if row["zone"] == "ablation": #int(row["id"]) <= 179 and
                ablation_cc_gain.append(float(row["mean_cc_gain"]))
                ablation_cc_gain_seis.append(float(row["mean_cross_gain"]))
                ablation_snr_power.append(float(row["mean_snr_power"]))
                #ablation_snr_variance.append(float(row["mean_snr_variance"]))
                #ablation_snr_rms.append(float(row["mean_snr_rms"]))
                #ablation_snr_absolute.append(float(row["mean_snr_absolut"]))
            if row["zone"] == "accumulation": #int(row["id"]) <= 126 and
                accumulation_cc_gain.append(float(row["mean_cc_gain"]))
                accumulation_cc_gain_seis.append(float(row["mean_cross_gain"]))
                accumulation_snr_power.append(float(row["mean_snr_power"]))
                #accumulation_snr_variance.append(float(row["mean_snr_variance"]))
                #accumulation_snr_rms.append(float(row["mean_snr_rms"]))
                #accumulation_snr_absolute.append(float(row["mean_snr_absolut"]))

    """ 
    print(len(ablation_cc_gain))
    print(len(ablation_cc_gain_seis))
    print(len(ablation_snr_power))
    print(len(ablation_snr_variance))
    print(len(ablation_snr_rms))
    print(len(accumulation_cc_gain))
    print(len(accumulation_cc_gain_seis))
    print(len(accumulation_snr_power))
    print(len(accumulation_snr_variance))
    print(len(accumulation_snr_rms))
    """

    """ Calculate MEDIAN Value """
    abl_lwc_median = np.median(ablation_cc_gain)
    abl_cc_seis_median = np.median(ablation_cc_gain_seis)
    abl_snr_power_median = np.median(ablation_snr_power)
    #abl_snr_variance_median = np.median(ablation_snr_variance)
    #abl_snr_rms_median = np.median(ablation_snr_rms)
    #abl_snr_absolute_median = np.median(ablation_snr_absolute)

    acc_lwc_median = np.median(accumulation_cc_gain)
    acc_cc_seis_median = np.median(accumulation_cc_gain_seis)
    acc_snr_power_median = np.median(accumulation_snr_power)
    #acc_snr_variance_median = np.median(accumulation_snr_variance)
    #acc_snr_rms_median = np.median(accumulation_snr_rms)
    #acc_snr_absolute_median = np.median(accumulation_snr_absolute)

    """ Print MEDIAN Values """

    print(experiment,  ": \n")
    print("Abl LWC: ", abl_lwc_median)
    print("Abl CCseis: ", abl_cc_seis_median)
    print("Abl SNR_POWER: ", abl_snr_power_median)
    #print("Abl SNR_VARIANCE: ", abl_snr_variance_median)
    #print("Abl SNR_RMS: ", abl_snr_rms_median)
    #print("Abl SNR_Absolute: ", abl_snr_absolute_median)
    print("\n")
    print("Acc LWC: ", acc_lwc_median)
    print("Acc CCseis:", acc_cc_seis_median)
    print("Acc SNR POWER: ", acc_snr_power_median)
    #print("Acc SNR VARIANCE: ", acc_snr_variance_median)
    #print("Acc SNR RMS: ", acc_snr_rms_median)
    #print("Acc SNR Absolute: ", acc_snr_absolute_median)
    print("\n \n \n")



    """ Plotting Data: 
    plt.plot(ablation_cc_gain, label = "ablation_cc_gain", marker="o", linestyle="")
    plt.plot(accumulation_cc_gain, label="accumulation_cc_gain", marker="o", linestyle="")
    plt.title("Experiment: " + experiment + "; LWC")
    plt.legend()
    plt.show()

    plt.plot(ablation_cc_gain_seis, label="ablation_cc_gain_seis", marker="o", linestyle="")
    plt.plot(accumulation_cc_gain_seis, label = "accumulation_cc_gain_seis", marker="o", linestyle="")
    plt.title("Experiment: " + experiment + "; CC Seis")
    plt.legend()
    plt.show()

    plt.plot(ablation_snr, label="ablation_cc_gain_seis", marker="o", linestyle="")
    plt.plot(accumulation_snr, label="accumulation_cc_gain_seis", marker="o", linestyle="")
    plt.title("Experiment: " + experiment + "; CC Seis")
    plt.legend()
    plt.show()
    """


