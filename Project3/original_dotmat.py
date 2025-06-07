import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def plot_originals(mat_path='dip_hw_3.mat'):
    """
    Load and display the two original RGB images stored in a .mat file.
    """
    data = loadmat(mat_path)
    d2a = data['d2a']
    d2b = data['d2b']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(d2a)
    axs[0].set_title('Original d2a')
    axs[0].axis('off')

    axs[1].imshow(d2b)
    axs[1].set_title('Original d2b')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig('original_images.png')

if __name__ == '__main__':
    plot_originals()
