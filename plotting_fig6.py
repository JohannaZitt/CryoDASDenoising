import csv
import os

import matplotlib.pyplot as plt
import numpy as np


"""

Here Figure 6 is generated. 

"""


#experiments = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical",
#               "05_combined200", "06_combined800", "07_retrained_combined200", "08_retrained_combined800",
#               "09_borehole_seismometer"]
experiments = ["07_retrained_combined200"]#, "07_retrained_combined200", "11_vanende", "12_vanende_finetuned_cryo", "13_isken_filter", "14_julius_filter"]
values_ablation_cc_gain = {}
values_ablation_cc_gain_seis = {}
values_accumulation_cc_gain = {}
values_accumulation_cc_gain_seis = {}



for experiment in experiments:

    """ Path to cc values """
    csv_path = os.path.join("experiments", experiment, "cc_evaluation_" + experiment[0:2] + ".csv")

    with open(csv_path, "r") as file:

        """ Read cc values """
        csv_reader = csv.DictReader(file)

        ablation_cc_gain = []
        ablation_cc_gain_seis = []
        accumulation_cc_gain = []
        accumulation_cc_gain_seis = []

        for row in csv_reader:

            if int(row["id"]) <= 179 and row["zone"] == "ablation":
                ablation_cc_gain.append(float(row["mean_cc_gain"]))
                ablation_cc_gain_seis.append(float(row["mean_cross_gain"]))
            if int(row["id"]) <= 126 and row["zone"] == "accumulation":
                accumulation_cc_gain.append(float(row["mean_cc_gain"]))
                accumulation_cc_gain_seis.append(float(row["mean_cross_gain"]))

    """ Calculate Mean Values for Plot """
    values_ablation_cc_gain[experiment[3:]] = np.mean(ablation_cc_gain)
    values_ablation_cc_gain_seis[experiment[3:]] = np.mean(ablation_cc_gain_seis)
    values_accumulation_cc_gain[experiment[3:]] = np.mean(accumulation_cc_gain)
    values_accumulation_cc_gain_seis[experiment[3:]] = np.mean(accumulation_cc_gain_seis)


    """ Calculate MEDIAN Value for Plot """


""" Generate Local Waveform Coherence Subplot via datatype "local" """
""" Generate Cross Correlation Subplot between DAS and seismometer data via datatype "seis" """
datatypes = ["local", "seis"]
for datatype in datatypes:

    # Data:
    labels = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical"
              , "Combined 200", "Combined 800", "Fine-Tuned\nCombined 200", "Fine-Tuned\nCombined 800", "Borehole"]

    if datatype == "local":
        acc_cc_gain = list(values_accumulation_cc_gain.values())
        abl_cc_gain = list(values_ablation_cc_gain.values())
        label = "Local Waveform Coherence Gain []"
    else:
        acc_cc_gain = list(values_accumulation_cc_gain_seis.values())
        abl_cc_gain = list(values_ablation_cc_gain_seis.values())
        label = "Cross Correlation Seismometer Gain []"

    """ Print Exact Values: """
    print(label)
    print("Labels", experiments)
    print("Accumulation: ", acc_cc_gain)
    print("Ablation: ", abl_cc_gain)

    """ Parameters """
    col_acc = "#008B93"
    col_abl = "#DF575F"
    width = 0.2  # depth of the bars
    x = np.arange(len(labels))  # x-Koordinaten for bars
    fontsize = 12
    gain = 2 #fontsize gain

    plt.figure(figsize=(13, 6))
    plt.grid(axis="y", zorder = 0)
    plt.bar(x - 0.5 * width, abl_cc_gain, width=width, label="Ablation", color=col_abl, zorder=3)
    plt.bar(x + 0.5 * width, acc_cc_gain, width=width, label="Accumulation", color=col_acc, zorder=3)

    plt.axhline(y=1, color="blue", zorder=4)
    plt.ylabel(label, fontsize=fontsize+gain)
    plt.xticks(x, labels, rotation=20, fontsize=fontsize+1)





    """ Save Plot """
    plt.tight_layout()
    plt.show()
    #plt.savefig("plots/fig_6_" + datatype + ".pdf", dpi=400)

