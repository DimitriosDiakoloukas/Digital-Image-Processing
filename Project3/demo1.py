from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from spectral_clustering import spectral_clustering

def main():
    data = loadmat("dip_hw_3.mat")
    d1a = data["d1a"]

    for k in [2, 3, 4]:
        labels = spectral_clustering(d1a, k)
        n_points = labels.shape[0]
        counts = np.bincount(labels.astype(int))
        print(f"\n=== demo1: k = {k} ===")
        print(" first 20 labels:", labels[:20])
        print(" cluster sizes:", counts)

        num_to_plot = min(20, n_points)
        plt.figure()
        plt.scatter(np.arange(num_to_plot), labels[:num_to_plot], s=20)
        plt.title(f"First {num_to_plot} labels (k={k})")
        plt.xlabel("Data Point Index")
        plt.ylabel("Cluster Label")
        plt.tight_layout()
        plt.savefig(f"demo1_k{k}_first_labels.png")
        plt.close()

        plt.figure()
        plt.bar(np.arange(len(counts)), counts, color='blue', alpha=0.7, width=0.4)
        plt.title(f"Cluster sizes (k={k})")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of points")
        plt.xticks(np.arange(len(counts)))
        plt.tight_layout()
        plt.savefig(f"demo1_k{k}_cluster_sizes.png")
        plt.close()

if __name__ == "__main__":
    main()
