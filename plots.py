import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Example aggregated metrics for English-only vs Multilingual
datasets = {
    "GT-Pairs": {
        "English": {"CER": [0.01005, 0.01055, 0.01379], "WER": [0.04904, 0.05126, 0.07958]},
        "Multilingual": {"CER": [1.19190, 1.19570, 1.19419], "WER": [2.72350, 2.73640, 2.76344]}
    },
    "BlurGT-Pairs": {
        "English": {"CER": [0.01697, 0.03674, 0.04574], "WER": [0.08483, 0.16871, 0.23490]},
        "Multilingual": {"CER": [1.17588, 1.19513, 1.19654], "WER": [2.73668, 2.79610, 2.85065]}
    },
    "iso_noise_folder": {
        "English": {"CER": [0.00946, 0.01110, 0.01094], "WER": [0.05096, 0.05291, 0.05871]},
        "Multilingual": {"CER": [1.18932, 1.18738, 1.19770], "WER": [2.72144, 2.73812, 2.76735]}
    },
    "PerspectiveGT-Pairs": {
        "English": {"CER": [0.01200, 0.01601, 0.01701], "WER": [0.06438, 0.07748, 0.08138]},
        "Multilingual": {"CER": [1.19408, 1.18780, 1.18667], "WER": [2.77462, 2.78798, 2.81399]}
    },
    "jpeg_folder": {
        "English": {"CER": [0.00951, 0.01142, 0.01438], "WER": [0.05251, 0.06660, 0.08835]},
        "Multilingual": {"CER": [1.18773, 1.19465, 1.19471], "WER": [2.72185, 2.74601, 2.75721]}
    },
    "ShadowedGT-Pairs": {
        "English": {"CER": [0.00971, 0.01125, 0.01288], "WER": [0.04674, 0.05536, 0.06575]},
        "Multilingual": {"CER": [1.19193, 1.19942, 1.20072], "WER": [2.73277, 2.75841, 2.76893]}
    },
}

conditions = ["Clean", "FGSM", "PGD"]
colors = ["#1f77b4", "#ff7f0e"]  # English, Multilingual
width = 0.35

for dataset_name, metrics in datasets.items():
    x = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CER subplot
    axes[0].bar(x - width/2, metrics["English"]["CER"], width, label="English-only", color=colors[0])
    axes[0].bar(x + width/2, metrics["Multilingual"]["CER"], width, label="Multilingual", color=colors[1])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions)
    axes[0].set_title("CER")
    axes[0].set_ylabel("CER")
    axes[0].legend()
    for i in range(len(conditions)):
        axes[0].text(x[i] - width/2, metrics["English"]["CER"][i] + 0.02*metrics["English"]["CER"][i],
                     f"{metrics['English']['CER'][i]:.3f}", ha='center', fontsize=9)
        axes[0].text(x[i] + width/2, metrics["Multilingual"]["CER"][i] + 0.02*metrics["Multilingual"]["CER"][i],
                     f"{metrics['Multilingual']['CER'][i]:.3f}", ha='center', fontsize=9)
    
    # WER subplot
    axes[1].bar(x - width/2, metrics["English"]["WER"], width, label="English-only", color=colors[0])
    axes[1].bar(x + width/2, metrics["Multilingual"]["WER"], width, label="Multilingual", color=colors[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions)
    axes[1].set_title("WER")
    axes[1].set_ylabel("WER")
    axes[1].legend()
    for i in range(len(conditions)):
        axes[1].text(x[i] - width/2, metrics["English"]["WER"][i] + 0.02*metrics["English"]["WER"][i],
                     f"{metrics['English']['WER'][i]:.3f}", ha='center', fontsize=9)
        axes[1].text(x[i] + width/2, metrics["Multilingual"]["WER"][i] + 0.02*metrics["Multilingual"]["WER"][i],
                     f"{metrics['Multilingual']['WER'][i]:.3f}", ha='center', fontsize=9)
    
    fig.suptitle(f"{dataset_name} — Baseline Disparity: English vs Multilingual")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{dataset_name}_CER_WER_disparity.png", dpi=200)
    plt.close()

print("Done! CER & WER disparity plots saved as PNG files.")
