import matplotlib.pyplot as plt


"""

Plotting model comparison as ...plot


"""


models = ["JI_earth", "JI_eartchcryo", "JI_cryoabl", "JI_cryoacc", "JI_cryocomb", "conservative", "AFK", "DASDL"]
LWC_abl = [2.2455, 1.9529, 1.5557, 1.8379, 1.8978, 1.0354, 1.1387, 5.6355]
LWC_acc = [2.7321, 1.9544, 1.6281, 2.1237, 2.0261, 1.0450, 1.0871, 5.9572]
CC_abl = [1.9675, 1.6696, 3.2889, 4.0322, 1.4221, 1.1452, 1.7831, 1.8879]
CC_acc = [1.5368, 1.2922, 2.6046, 4.0088, 1.1843, 1.1189, 1.3351, 1.4968]
SNR_abl = [0.9404, 0.9937, 0.6505, 0.6409, 1.0119, 1.0029, 0.9765, 0.5610]
SNR_acc = [0.9520, 0.9678, 1.8635, 2.8002, 1.0008, 0.9916, 1.1135, 0.5330]
ERT = [10.7288, 10.3046, 11.2729, 11.2185, 11.6818, 0.4746, 0.1058, 37.5446]



# Create a figure with 4 subplots
plt.figure(figsize=(16, 12))

# LWC Subplot
plt.subplot(2, 2, 1)
plt.plot(models, LWC_abl, marker="o", linestyle='--', label='Ablation', markersize = 8)
plt.plot(models, LWC_acc, marker="o", linestyle='--', label='Accumulation', markersize = 8)
plt.title('Local Waveform Coherence (LWC)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('LWC Values')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# CC Subplot
plt.subplot(2, 2, 2)
plt.plot(models, CC_abl, marker='o', linestyle='--', label='Ablation', markersize = 8)
plt.plot(models, CC_acc, marker='o', linestyle='--', label='Accumulation', markersize = 8)
plt.title('Cross Correlation with Seismometer (CC)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('CC Values')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# SNR Subplot
plt.subplot(2, 2, 3)
plt.plot(models, SNR_abl, marker='o', linestyle='--', label='Ablation', markersize = 8)
plt.plot(models, SNR_acc, marker='o', linestyle='--', label='Accumulation', markersize = 8)
plt.title('Signal to Noise Ratio (SNR)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('SNR Values')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# ERT Subplot
plt.subplot(2, 2, 4)
plt.plot(models, ERT, marker='o', linestyle='--', markersize = 8)
plt.title('Estimated Response Time (ERT)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('ERT Values')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and add overall title
plt.tight_layout()

# Show the plot
plt.show()
