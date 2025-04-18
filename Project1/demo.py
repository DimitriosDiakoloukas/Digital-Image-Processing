# demo.py
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hist_modif import perform_hist_eq, perform_hist_matching
from hist_utils import calculate_hist_of_img


def _read_grayscale(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def _hist_data(img: np.ndarray):
    h = calculate_hist_of_img(img, True)           # relative frequencies
    k = np.array(sorted(h.keys()), dtype=float)
    v = np.array([h[x] for x in k], dtype=float)
    return k, v


def _plot_pair(img_a: np.ndarray, img_b: np.ndarray, titles: List[str], fname: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(titles[0])
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(titles[1])
    axes[0, 1].axis("off")

    k, v = _hist_data(img_a)
    axes[1, 0].bar(k, v, width=0.003)
    axes[1, 0].set_title("Histogram")

    k, v = _hist_data(img_b)
    axes[1, 1].bar(k, v, width=0.003)
    axes[1, 1].set_title("Histogram")

    for ax in axes[1]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, v.max() * 1.05)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Rel. freq.")

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def main():
    modes = ["greedy", "non-greedy", "post-disturbance"]
    root = pathlib.Path(__file__).parent
    img_in = _read_grayscale(root / "input_img.jpg")
    img_ref = _read_grayscale(root / "ref_img.jpg")

    for m in modes:
        eq = perform_hist_eq(img_in, m)
        _plot_pair(img_in, eq, ["Original", f"Equalised ({m})"], root / f"eq_{m}.png")

    for m in modes:
        matched = perform_hist_matching(img_in, img_ref, m)
        _plot_pair(img_in, matched, ["Original", f"Matched ({m})"], root / f"match_{m}.png")


if __name__ == "__main__":
    main()
