import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Data from the evaluations

datasets = {
    "GT-pairs": {
        "CER": [0.010050449783562562, 0.010548854590305272, 0.013785147762568782],
        "WER": [0.04903968838179363, 0.051255455071244536, 0.07958480116374854],
    },
    "BlurGT-Pairs": {
        "CER": [0.01697240816694629, 0.03673525551776237, 0.0457400997135171],
        "WER": [0.0848285924601714, 0.16871330423962, 0.2348978945031576],
    },
    "iso_noise_folder": {
        "CER": [0.009457583243376548, 0.011096172427127637, 0.010942323137648558],
        "WER": [0.050960968855705696, 0.05291331475542, 0.058711463974621854],
    },
    "PerspectiveGT-Pairs": {
        "CER": [0.012003380583986574, 0.016009341623903692, 0.017014107656323083],
        "WER": [0.06437763990395566, 0.07748277307487833, 0.08138257940889516],
    },
    "jpeg_folder": {
        "CER": [0.00951004540699576, 0.011421752747423098, 0.014377417752238246],
        "WER": [0.0525125751441541, 0.06660006660006661, 0.08834647808332019],
    },
    "ShadowedGT-Pairs": {
        "CER": [0.00971289078122896, 0.011246542222235778, 0.012880766995248762],
        "WER": [0.046741767004924904, 0.05535946509630721, 0.06574936028883396],
    },
}

dataset_name = list(datasets.keys())
color_map = cm.get_cmap("tab10", len(dataset_name))
dataset_colors = {
        name: color_map(i) for i, name in enumerate(dataset_name)
}

conditions = ["Clean", "FGSM", "PGD"]

# Plotting

for dataset_name, metrics in datasets.items():
    for metric_name, values in metrics.items():

        plt.figure(figsize=(8, 5))

        x = np.arange(len(conditions))
        plt.bar(x, values, color = dataset_colors[dataset_name])

        plt.xticks(x, conditions)
        plt.title(f"{dataset_name} — {metric_name}")
        plt.ylabel(metric_name)
        plt.xlabel("Condition")

        for i, v in enumerate(values):
            plt.text(i, v + (v * 0.02), f"{v:.4f}", ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{dataset_name}_{metric_name}.png", dpi=200)
        plt.close()

print("Done! Plots saved as PNG files.")
