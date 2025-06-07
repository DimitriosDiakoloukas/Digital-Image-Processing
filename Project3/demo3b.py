import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from normalized_cuts import n_cuts, calculate_n_cut_value
from image_graph import image_to_graph

def demo3b():
    data = loadmat('dip_hw_3.mat')
    for varname in ['d2a', 'd2b']:
        img = data[varname]
        M, N, _ = img.shape
        W = image_to_graph(img)
        print(f"\n=== Demo3b: one-step (k=2) n_cuts on {varname} ===")
        labels = n_cuts(W, 2)
        counts = np.bincount(labels.astype(int))
        print(f"{varname}, cluster sizes: {counts}")
        ncut_val = calculate_n_cut_value(W, labels)
        print(f"Ncut value for the split: {ncut_val:.4f}")
        seg = labels.reshape(M, N)
        plt.figure(figsize=(4, 4))
        plt.imshow(seg, cmap='jet')
        plt.title(f"One-step n_cuts on {varname}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{varname}_one_step_n_cuts.png")

if __name__ == '__main__':
    demo3b()
