from typing import Tuple, Optional
import numpy as np


def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: Optional[np.ndarray] = None, mask_origin: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform 2-D linear convolution of an image with an FIR mask.

    Parameters
    ----------
    in_img_array : np.ndarray (float)
        Input grayscale image, shape (M, N).
    h : np.ndarray (float)
        Convolution mask (kernel), shape (P, Q).
    in_origin : np.ndarray, optional (int, 2-elements)
        Position [row, col] of (0,0) inside the input image.
        Default is [0,0].
    mask_origin : np.ndarray, optional (int, 2-elements)
        Position [row, col] of (0,0) inside the mask.
        Default is the mask centre [P//2, Q//2].

    Returns
    -------
    out_img_array : np.ndarray (float)
        Full size convolution result, shape (M+P-1, N+Q-1).
    out_origin : np.ndarray (int, 2-elements)
        Position [row, col] of (0,0) inside the output image.
    """
    if in_img_array.ndim != 2 or h.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays")

    # origin in the input image
    if in_origin is None:
        in_origin = np.array([0, 0], dtype=int)
    else:
        in_origin = np.asarray(in_origin, dtype=int).ravel()
        if in_origin.size != 2:
            raise ValueError("in_origin must have two values [row, col]")

    # origin in the mask
    if mask_origin is None:
        mask_origin = np.array([h.shape[0] // 2, h.shape[1] // 2], dtype=int)
    else:
        mask_origin = np.asarray(mask_origin, dtype=int).ravel()
        if mask_origin.size != 2:
            raise ValueError("mask_origin must have two values [row, col]")

    # ---------- build flipped mask ----------
    h_flipped = np.flip(h, axis=(0, 1))

    # ---------- zero‑pad input ----------
    pad_rows = h.shape[0] - 1
    pad_cols = h.shape[1] - 1
    padded = np.pad(
        in_img_array,
        ((pad_rows, pad_rows), (pad_cols, pad_cols)),
        mode="constant",
        constant_values=0.0,
    )

    # ---------- create output ----------
    out_shape = (
        in_img_array.shape[0] + h.shape[0] - 1,
        in_img_array.shape[1] + h.shape[1] - 1,
    )
    out_img_array = np.zeros(out_shape, dtype=float)

    # ---------- slide mask ----------
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            region = padded[i : i + h.shape[0], j : j + h.shape[1]]
            out_img_array[i, j] = np.sum(region * h_flipped)

    # ---------- origin in output ----------
    out_origin = in_origin + mask_origin

    return out_img_array, out_origin
