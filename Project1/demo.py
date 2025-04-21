import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hist_modif import perform_hist_eq, perform_hist_matching
from hist_utils import calculate_hist_of_img


def _read_grayscale(path: str) -> np.ndarray:
    """Load image, keep luminance only, rescale to [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def _hist_data(img: np.ndarray):
    """Return (sorted_levels, relative_frequencies)."""
    h = calculate_hist_of_img(img, return_normalized=True)
    levels = np.array(sorted(h.keys()), dtype=float)
    freqs = np.array([h[x] for x in levels], dtype=float)
    return levels, freqs


def _plot_pair(img_a: np.ndarray, img_b: np.ndarray, titles: List[str], fname: pathlib.Path):
    """Plot two images and their histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(titles[0]); axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(titles[1]); axes[0, 1].axis("off")

    k_a, v_a = _hist_data(img_a)
    mask_a = v_a > 0
    k_a, v_a = k_a[mask_a], v_a[mask_a]

    bar_w_a = np.diff(k_a).mean() if len(k_a) > 1 else 0.002
    axes[1, 0].bar(k_a, v_a, width=bar_w_a)
    axes[1, 0].set_title("Histogram")

    margin = 0.05
    x_left_a  = max(0.0, k_a.min() - margin)
    x_right_a = min(1.0, k_a.max() + margin)
    axes[1, 0].set_xlim(x_left_a, x_right_a)
    axes[1, 0].set_ylim(0, v_a.max() * 1.05)
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Rel. freq.")

    k_b, v_b = _hist_data(img_b)
    mask_b = v_b > 0
    k_b, v_b = k_b[mask_b], v_b[mask_b]

    bar_w_b = np.diff(k_b).mean() if len(k_b) > 1 else 0.002
    axes[1, 1].bar(k_b, v_b, width=bar_w_b)
    axes[1, 1].set_title("Histogram")

    x_left_b  = max(0.0, k_b.min() - margin)
    x_right_b = min(1.0, k_b.max() + margin)
    axes[1, 1].set_xlim(x_left_b, x_right_b)
    axes[1, 1].set_ylim(0, v_b.max() * 1.05)
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Rel. freq.")

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
