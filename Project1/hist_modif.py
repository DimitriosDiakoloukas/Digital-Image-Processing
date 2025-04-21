import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform

_VALID_MODES = {"greedy", "non-greedy", "post-disturbance"}


def _desired_counts(hist_ref: Dict[float, float], total: int):
    """Convert a target histogram (relative frequencies) into absolute counts.

    Rounds ideal counts down, then distributes the remaining pixels to the
    bins with the largest fractional remainders.

    Parameters
    ----------
    hist_ref : Dict[float, float]
        Target histogram.
    total : int
        Total number of pixels in the image to match.

    Returns
    -------
    g : np.ndarray
        Sorted output intensity levels (gi).
    c : np.ndarray
        Integer counts for each level, summing to total.
    """
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


def _greedy_map(img_hist: Dict[float, int], g: np.ndarray, desired: np.ndarray) -> Dict[float, float]:
    """Greedy mapping: assign input levels to output bins sequentially.

    Adds each fᵢ to gⱼ until the total assigned to gj first reaches or
    exceeds desired[j], then proceeds to gj+1.

    Parameters
    ----------
    img_hist : Dict[float, int]
        Absolute histogram of input image.
    g : np.ndarray
        Sorted output levels.
    desired : np.ndarray
        Target counts for each gj.

    Returns
    -------
    Dict[float, float]
        Mapping fi → gj.
    """
    f = np.array(list(img_hist.keys()), dtype=float)
    cnt = np.array(list(img_hist.values()), dtype=int)
    order = np.argsort(f)
    f, cnt = f[order], cnt[order]

    mapping: Dict[float, float] = {}
    gi = 0                    
    filled = 0               

    for fi, c in zip(f, cnt):
        mapping[fi] = float(g[gi])
        filled += c

        if filled >= desired[gi] and gi + 1 < len(g):
            gi += 1
            filled = 0        
    return mapping


def _non_greedy_map(img_hist: Dict[float, int], g: np.ndarray, desired: np.ndarray) -> Dict[float, float]:
    """Non-greedy mapping: minimise mismatch by comparing deficiencies.

    For each output bin gj, add input levels fi until the deficiency would
    increase significantly by doing so. Avoids overshooting.

    Parameters
    ----------
    img_hist : Dict[float, int]
        Absolute histogram of input image.
    g : np.ndarray
        Sorted output levels.
    desired : np.ndarray
        Target counts for each gj.

    Returns
    -------
    Dict[float, float]
        Mapping fi → gj.
    """
    f = np.array(list(img_hist.keys()), dtype=float)
    cnt = np.array(list(img_hist.values()), dtype=int)
    order = np.argsort(f)
    f, cnt = f[order], cnt[order]

    mapping: Dict[float, float] = {}
    i = 0 

    for j, target in enumerate(desired):
        taken = 0 

        if i < len(f):
            mapping[f[i]] = float(g[j])
            taken += cnt[i]
            i += 1

        while i < len(f):
            deficiency = target - taken
            if deficiency >= cnt[i] / 2:
                mapping[f[i]] = float(g[j])
                taken += cnt[i]
                i += 1
            else:
                break

    while i < len(f):
        mapping[f[i]] = float(g[-1])
        i += 1

    return mapping


def _post_disturbance(img_array: np.ndarray, g: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Post-disturbance method: break quantisation, then sort and assign.

    Adds uniform noise to each pixel (to avoid ties), then sorts and assigns
    output levels exactly according to the desired distribution.

    Parameters
    ----------
    img_array : np.ndarray
        Grayscale image, shape HxW.
    g : np.ndarray
        Output intensity levels (sorted).
    desired : np.ndarray
        Number of pixels to assign to each output level.

    Returns
    -------
    np.ndarray
        Image of same shape with reassigned intensity levels.
    """
    flat = img_array.ravel()

    u = np.unique(flat)
    d = np.median(np.diff(u)) if u.size > 1 else 1.0

    noise = np.random.uniform(-d / 2, d / 2, flat.size)
    disturbed = flat + noise

    order = np.argsort(disturbed)
    out = np.empty_like(flat)

    start = 0
    for level, cnt in zip(g, desired):
        end = start + cnt
        out[order[start:end]] = level
        start = end

    return out.reshape(img_array.shape)



def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict[float, float], mode: str) -> np.ndarray:
    """Modify the histogram of an image using one of the three methods.

    Parameters
    ----------
    img_array : np.ndarray
        Input grayscale image.
    hist_ref : Dict[float, float]
        Reference histogram (values are relative frequencies).
    mode : str
        One of: "greedy", "non-greedy", or "post-disturbance".

    Returns
    -------
    np.ndarray
        Transformed image.
    """
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")
    g, desired = _desired_counts(hist_ref, img_array.size)

    mode = mode.lower()
    if mode in {"greedy", "non-greedy"}:
        img_hist = calculate_hist_of_img(img_array, False)
        if mode == "greedy":
            mapping = _greedy_map(img_hist, g, desired)
        else:
            mapping = _non_greedy_map(img_hist, g, desired)
        return apply_hist_modification_transform(img_array, mapping)

    if mode == "post-disturbance":
        return _post_disturbance(img_array, g, desired)

    raise ValueError("mode must be 'greedy', 'non-greedy', or 'post-disturbance'")


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """Histogram equalisation to uniform distribution.

    Parameters
    ----------
    img_array : np.ndarray
        Input grayscale image.
    mode : str
        Modification method: "greedy", "non-greedy", or "post-disturbance".

    Returns
    -------
    np.ndarray
        Equalised image.
    """
    if img_array.ndim != 2:
        raise ValueError("img_array must be 2D")
    mode = mode.lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}")

    Lg = 256
    g_min, g_max = float(img_array.min()), float(img_array.max())
    g_levels = np.linspace(g_min, g_max, Lg)
    hist_ref: Dict[float, float] = {float(g): 1.0 / Lg for g in g_levels}

    return perform_hist_modification(img_array, hist_ref, mode)


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    """Histogram matching to the distribution of another image.

    Parameters
    ----------
    img_array : np.ndarray
        Input image to be transformed.
    img_array_ref : np.ndarray
        Reference image whose histogram should be matched.
    mode : str
        Modification method: "greedy", "non-greedy", or "post-disturbance".

    Returns
    -------
    np.ndarray
        Transformed image with matched histogram.
    """
    if img_array.ndim != 2 or img_array_ref.ndim != 2:
        raise ValueError("both images must be 2D")
    mode = mode.lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}")

    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    return perform_hist_modification(img_array, hist_ref, mode)