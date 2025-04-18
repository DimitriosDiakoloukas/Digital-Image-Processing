import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform


def _desired_counts(hist_ref: Dict[float, float], total: int):
    g = np.array(list(hist_ref.keys()), dtype=float)
    p = np.array(list(hist_ref.values()), dtype=float)
    p /= p.sum()
    ideal = p * total
    c = np.floor(ideal).astype(int)
    diff = total - c.sum()
    if diff:
        frac = ideal - c
        c[np.argsort(frac)[-diff:]] += 1
    i = np.argsort(g)
    return g[i], c[i]


def _greedy_map(img_hist: Dict[float, int], g: np.ndarray, desired: np.ndarray):
    f = np.array(list(img_hist.keys()), dtype=float)
    cnt = np.array(list(img_hist.values()), dtype=int)
    i = np.argsort(f)
    f, cnt = f[i], cnt[i]
    m = {}
    j, remain = 0, desired[0]
    for x, c in zip(f, cnt):
        m[x] = g[j]
        remain -= c
        if remain <= 0 and j + 1 < len(g):
            j += 1
            remain = desired[j]
    return m


def _non_greedy_map(img_hist: Dict[float, int], g: np.ndarray, desired: np.ndarray):
    f = np.array(list(img_hist.keys()), dtype=float)
    cnt = np.array(list(img_hist.values()), dtype=int)
    i = np.argsort(f)
    f, cnt = f[i], cnt[i]
    m, k = {}, 0
    for j, need in enumerate(desired):
        taken, first = 0, True
        while k < len(f):
            x, c = f[k], cnt[k]
            if first:
                m[x] = g[j]
                taken += c
                k += 1
                first = False
                continue
            if need - taken >= c / 2:
                m[x] = g[j]
                taken += c
                k += 1
            else:
                break
    while k < len(f):
        m[f[k]] = g[-1]
        k += 1
    return m


def _post_disturbance(img_array: np.ndarray, g: np.ndarray, desired: np.ndarray):
    flat = img_array.ravel()
    u = np.unique(flat)
    d = float(u[1] - u[0]) if u.size > 1 else 1.0
    disturbed = flat + np.random.uniform(-d / 2, d / 2, flat.shape[0])
    order = np.argsort(disturbed)
    out = np.empty_like(flat)
    start = 0
    for level, cnt in zip(g, desired):
        end = start + cnt
        out[order[start:end]] = level
        start = end
    return out.reshape(img_array.shape)


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict[float, float], mode: str) -> np.ndarray:
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2‑D")
    g, desired = _desired_counts(hist_ref, img_array.size)
    img_hist = calculate_hist_of_img(img_array, False)
    if mode == "greedy":
        m = _greedy_map(img_hist, g, desired)
        return apply_hist_modification_transform(img_array, m)
    if mode == "non-greedy":
        m = _non_greedy_map(img_hist, g, desired)
        return apply_hist_modification_transform(img_array, m)
    if mode == "post-disturbance":
        return _post_disturbance(img_array, g, desired)
    raise ValueError("mode must be 'greedy', 'non-greedy', or 'post-disturbance'")


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    g = np.linspace(img_array.min(), img_array.max(), 256)
    hist_ref = {float(x): 1 / 256 for x in g}
    return perform_hist_modification(img_array, hist_ref, mode)


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    ref_hist = calculate_hist_of_img(img_array_ref, True)
    return perform_hist_modification(img_array, ref_hist, mode)
