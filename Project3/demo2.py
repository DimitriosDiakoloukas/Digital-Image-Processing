import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from image_graph import image_to_graph
from spectral_clustering import spectral_clustering

def main():
    data = loadmat("dip_hw_3.mat")
    for varname in ["d2a", "d2b"]:
        img = data[varname]           # shape (M,N,3)
        M, N, _ = img.shape

        print(f"\n=== demo2: image {varname} ({M}x{N}) ===")
        W = image_to_graph(img)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, k in zip(axes, [2, 3, 4]):
            labels = spectral_clustering(W, k)
            seg = labels.reshape(M, N)
            ax.imshow(seg, cmap="jet")
            ax.set_title(f"k = {k}")
            ax.axis("off")

        plt.suptitle(f"Spectral Clustering on {varname}")
        plt.tight_layout()
        plt.savefig(f"{varname}_segmentation.png")
        print("Saved segmentation images for", varname)

if __name__ == "__main__":
    main()
