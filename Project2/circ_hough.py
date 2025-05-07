from typing import Tuple
import numpy as np


def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_min: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular Hough transform.

    Parameters
    ----------
    in_img_array : np.ndarray (int)
        Binary edge map, shape (H, W), values {0,1}.
    R_max : float
        Maximum circle radius to consider.
    dim : np.ndarray (int, len=3)
        Number of bins [Nx, Ny, Nr] for the Hough space.
    V_min : int
        Minimum votes for a bin to be kept.

    Returns
    -------
    centers : np.ndarray (float, shape (K,2))
        Detected centre coordinates [row, col].
    radii   : np.ndarray (float, shape (K,))
        Detected radii.
    """
    if in_img_array.ndim != 2:
        raise ValueError("in_img_array must be 2D")
    if dim.size != 3:
        raise ValueError("dim must have three elements [Nx, Ny, Nr]")
    if R_max <= 0 or V_min <= 0:
        raise ValueError("R_max and V_min must be positive values")

    H, W = in_img_array.shape
    Nx, Ny, Nr = dim.astype(int)

    # ---- set up Hough accumulator ----
    acc = np.zeros((Ny, Nx, Nr), dtype=int)

    # Bin sizes in parameter space
    dx = W / Nx
    dy = H / Ny
    r_vals = (np.arange(Nr) + 0.5) * (R_max / Nr)  # bin centres

    # Edge pixel coordinates (row, col)
    edge_points = np.argwhere(in_img_array > 0)

    # Pre‑compute a coarse circle perimeter for each r
    theta = np.deg2rad(np.arange(0, 360, 10))  # 36 directions
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for row, col in edge_points:
        for ir, r in enumerate(r_vals):
            # Potential centres for this (row, col, r)
            cx = col - r * cos_t
            cy = row - r * sin_t

            # Keep centres that fall inside the image
            mask = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            cx_valid = cx[mask]
            cy_valid = cy[mask]

            # Map centre coords to accumulator indices
            ia = np.floor(cx_valid / dx).astype(int)
            ib = np.floor(cy_valid / dy).astype(int)

            for a_bin, b_bin in zip(ia, ib):
                acc[b_bin, a_bin, ir] += 1

    # ---- threshold the accumulator ----
    bins = np.argwhere(acc >= V_min)
    K = len(bins)
    centers = np.zeros((K, 2), dtype=float)  # [row, col]
    radii = np.zeros(K, dtype=float)

    for k, (b_bin, a_bin, r_bin) in enumerate(bins):
        # Map bin indices back to continuous parameters
        centers[k, 1] = (a_bin + 0.5) * dx  # col (x)
        centers[k, 0] = (b_bin + 0.5) * dy  # row (y)
        radii[k] = r_vals[r_bin]

    return centers, radii
