from typing import Tuple
import numpy as np
from fir_conv import fir_conv


def _zero_crossings(img: np.ndarray) -> np.ndarray:
    """
    Mark pixels whose 3x3 neighbourhood contains both
    positive and negative values.

    Parameters
    ----------
    img : np.ndarray
        Filtered (LoG) image.

    Returns
    -------
    out : np.ndarray (int)
        Binary edge map (1=edge).
    """
    rows, cols = img.shape
    out = np.zeros((rows, cols), dtype=int)

    # Ignore the outer one‑pixel border
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            patch = img[r - 1 : r + 2, c - 1 : c + 2]
            if patch.min() < 0 and patch.max() > 0:
                out[r, c] = 1
    return out


def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    """
    LoG edge detection with zero crossings.

    Parameters
    ----------
    in_img_array : np.ndarray (float)
        Grayscale image with values in [0,1].

    Returns
    -------
    out_img_array : np.ndarray (int)
        Binary edge map, same shape, values {0,1}.
    """
    if in_img_array.ndim != 2:
        raise ValueError("in_img_array must be 2D")

    # ---- 5×5 LoG mask from the assignment ----
    h_log = np.array(
        [[ 0,  0, -1,  0,  0],
         [ 0, -1, -2, -1,  0],
         [-1, -2, 16, -2, -1],
         [ 0, -1, -2, -1,  0],
         [ 0,  0, -1,  0,  0]],
        dtype=float,
    )
    mask_origin = np.array([2, 2], dtype=int)  # centre of 5×5
    in_origin = np.array([0, 0], dtype=int)

    # ---- convolution (full) ----
    conv_full, out_origin = fir_conv(
        in_img_array,
        h_log,
        in_origin=in_origin,
        mask_origin=mask_origin,
    )

    # ---- crop back to original size ----
    r0, c0 = out_origin
    r1 = r0 + in_img_array.shape[0]
    c1 = c0 + in_img_array.shape[1]
    log_img = conv_full[r0:r1, c0:c1]

    # ---- zero‑crossing detection ----
    out_img_array = _zero_crossings(log_img)

    return out_img_array
