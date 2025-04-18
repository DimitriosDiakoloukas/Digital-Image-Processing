import numpy as np
from typing import Dict
from PIL import Image
import numpy as np

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict[float, float]:
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")
    flat = img_array.ravel()
    values, counts = np.unique(flat, return_counts=True)
    if return_normalized:
        counts = counts / flat.size
    return {float(v): float(c) for v, c in zip(values, counts)}

def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict[float, float]) -> np.ndarray:
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")
    keys = np.fromiter(modification_transform.keys(), dtype=float)
    vals = np.fromiter(modification_transform.values(), dtype=float)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    vals_sorted = vals[order]
    flat = img_array.ravel()
    idx = np.searchsorted(keys_sorted, flat)
    if np.any(keys_sorted[idx] != flat):
        raise KeyError("modification_transform does not cover all input levels")
    out = vals_sorted[idx]
    return out.reshape(img_array.shape)


# filename = "img1.jpg"
# img = Image.open(fp=filename)
# bw_img = img.convert("L")
# img_array = np.array(bw_img).astype(float) / 255.0

# import numpy as np
# from typing import Dict


# def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict[float, float]:
#     if img_array.ndim != 2:
#         raise ValueError("img_array must be 2‑D")
#     flat = img_array.ravel()
#     vals, counts = np.unique(flat, return_counts=True)
#     if return_normalized:
#         counts = counts / flat.size
#     return {float(v): float(c) for v, c in zip(vals, counts)}


# def apply_hist_modification_transform(
#     img_array: np.ndarray, modification_transform: Dict[float, float]
# ) -> np.ndarray:
#     if img_array.ndim != 2:
#         raise ValueError("img_array must be 2‑D")
#     keys = np.fromiter(modification_transform.keys(), dtype=float)
#     vals = np.fromiter(modification_transform.values(), dtype=float)
#     order = np.argsort(keys)
#     keys, vals = keys[order], vals[order]
#     flat = img_array.ravel()
#     idx = np.searchsorted(keys, flat)
#     if np.any(keys[idx] != flat):
#         raise KeyError("modification_transform lacks some input levels")
#     out = vals[idx]
#     return out.reshape(img_array.shape)
