import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from n_cuts import n_cuts_recursive
from spectral_clustering import spectral_clustering
from image_to_graph import image_to_graph

def demo3c():
    data = loadmat('dip_hw_3.mat')
    T1 = 5.0 
    T2 = 0.20

    # Experimenting with different T1 and T2 values
    # T1 = 1000 # now 2500 > 1000, so it will bisect the root d2a, recursive cluster sizes: [2500]
    # T2 = np.inf
    for varname in ['d2a', 'd2b']:
        img = data[varname]
        M, N, _ = img.shape
        W = image_to_graph(img)
        print(f"\n=== Demo3c: recursive n_cuts on {varname}, T1={T1}, T2={T2} ===")
        labels_rec = n_cuts_recursive(W, T1, T2)
        counts_rec = np.bincount(labels_rec.astype(int))
        print(f"{varname}, recursive cluster sizes: {counts_rec}")

        # For comparison, also run spectral clustering at k=2 and k=3
        labels_sc2 = spectral_clustering(W, 2)
        labels_sc3 = spectral_clustering(W, 3)
        counts_sc2 = np.bincount(labels_sc2.astype(int))
        counts_sc3 = np.bincount(labels_sc3.astype(int))
        print(f"{varname}, spectral k=2 sizes: {counts_sc2}")
        print(f"{varname}, spectral k=3 sizes: {counts_sc3}")

        # Plot recursive vs spectral
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(labels_rec.reshape(M, N), cmap='jet')
        axes[0].set_title('recursive n_cuts')
        axes[0].axis('off')

        axes[1].imshow(labels_sc2.reshape(M, N), cmap='jet')
        axes[1].set_title('spectral k=2')
        axes[1].axis('off')

        axes[2].imshow(labels_sc3.reshape(M, N), cmap='jet')
        axes[2].set_title('spectral k=3')
        axes[2].axis('off')

        plt.suptitle(f'Comparison on {varname}')
        plt.tight_layout()
        plt.savefig(f"{varname}_comparison.png")
        plt.close()

if __name__ == '__main__':
    demo3c()
