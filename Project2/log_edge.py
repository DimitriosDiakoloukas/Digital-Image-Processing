import numpy as np
from typing import Tuple
from fir_conv import fir_conv


def _zero_crossings(img: np.ndarray) -> np.ndarray:
    rows, cols = img.shape
    out = np.zeros((rows, cols), dtype=int)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            patch = img[r - 1:r + 2, c - 1:c + 2]
            if patch.min() < 0 and patch.max() > 0:
                out[r, c] = 1
    return out


def _generate_log_kernel(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    # Size: 6σ covers most of the Gaussian – force to be odd
    k = int(np.ceil(3 * sigma))
    x1 = np.arange(-k, k + 1)
    x2 = np.arange(-k, k + 1)
    X1, X2 = np.meshgrid(x1, x2)

    # Equation (6) from the slide
    factor = -1 / (np.pi * sigma**4)
    r_squared = X1**2 + X2**2
    log_kernel = factor * (1 - r_squared / (2 * sigma**2)) * np.exp(-r_squared / (2 * sigma**2))

    return log_kernel, np.array([k, k], dtype=int)


def log_edge(in_img_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if in_img_array.ndim != 2:
        raise ValueError("in_img_array must be 2D")

    # Generate LoG kernel from the exact formula (6)
    h_log, mask_origin = _generate_log_kernel(sigma)

    # Perform convolution
    conv_full, out_origin = fir_conv(
        in_img_array,
        h_log,
        in_origin=np.array([0, 0], dtype=int),
        mask_origin=mask_origin,
    )

    # Crop back to input shape
    r0, c0 = out_origin
    r1 = r0 + in_img_array.shape[0]
    c1 = c0 + in_img_array.shape[1]
    log_img = conv_full[r0:r1, c0:c1]

    # Detect edges via zero crossings
    return _zero_crossings(log_img)
