import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from normalized_cuts import n_cuts
from image_graph import image_to_graph

def demo3a():
    data = loadmat('dip_hw_3.mat')
    for varname in ['d2a', 'd2b']:
        img = data[varname]
        M, N, _ = img.shape
        W = image_to_graph(img)
        print(f"\n=== Demo3a: non-recursive n_cuts on {varname} ===")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, k in zip(axes, [2, 3, 4]):
            labels = n_cuts(W, k)
            counts = np.bincount(labels.astype(int))
            print(f"{varname}, k={k}, cluster sizes: {counts}")
            seg = labels.reshape(M, N)
            ax.imshow(seg, cmap='jet')
            ax.set_title(f"k = {k}")
            ax.axis('off')
        plt.suptitle(f'n_cuts (k=2,3,4) on {varname}')
        plt.tight_layout()
        plt.savefig(f"{varname}_n_cuts.png")
        plt.close()
        
if __name__ == '__main__':
    demo3a()
