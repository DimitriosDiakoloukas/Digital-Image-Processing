import numpy as np
from scipy.io import loadmat
from spectral_clustering import spectral_clustering

def main():
    data = loadmat("dip_hw_3.mat")
    d1a = data["d1a"]

    for k in [2, 3, 4]:
        labels = spectral_clustering(d1a, k)
        counts = np.bincount(labels.astype(int))
        print(f"\n=== demo1: k = {k} ===")
        print(" first 20 labels:", labels[:20])
        print(" cluster sizes:", counts)

if __name__ == "__main__":
    main()
