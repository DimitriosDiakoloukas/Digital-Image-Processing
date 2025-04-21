import numpy as np
from typing import Dict
import numpy as np

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict[float, float | int]:
    """Return the histogram of a 2D grayscale image.

    Parameters
    ----------
    img_array : np.ndarray
        HxW image with samples in [0,1].
    return_normalized : bool
        True -> values are relative frequencies (sum to1).  
        False -> values are absolute counts (integers).

    Raises
    ------
    ValueError
        If img_array is not two dimensional.
    """
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")

    flat = img_array.ravel()
    values, counts = np.unique(flat, return_counts=True)

    if return_normalized:
        counts = counts / flat.size
        return {float(v): float(c) for v, c in zip(values, counts)}
    else:
        return {float(v): int(c) for v, c in zip(values, counts)}


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict[float, float]) -> np.ndarray:
    """Apply a level to level mapping to every pixel of an image.

    Parameters
    ----------
    img_array : np.ndarray
        HxW grayscale image.
    modification_transform : Dict[float, float]
        Dict that maps each input intensity level fᵢ to an
        output level gj (must cover all levels present in img_array).

    Returns
    -------
    np.ndarray
        Image of the same shape with transformed intensities.

    Raises
    ------
    ValueError
        If img_array is not two dimensional.
    KeyError
        If the mapping lacks some intensity levels that appear in img_array.
    """
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")

    keys = np.fromiter(modification_transform.keys(), dtype=float)
    vals = np.fromiter(modification_transform.values(), dtype=float)
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]

    flat = img_array.ravel()
    idx = np.searchsorted(keys, flat)

    if np.any(keys[idx] != flat):
        missing = np.unique(flat[keys[idx] != flat])[:5]
        raise KeyError(f"The mapping does not cover intensity levels such as {missing}")

    out = vals[idx].astype(float, copy=False) 
    return out.reshape(img_array.shape)
