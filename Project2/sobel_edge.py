import numpy as np
from fir_conv import fir_conv


def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    """
    Detect binary edges with the Sobel operator.

    Parameters
    ----------
    in_img_array : np.ndarray (float)
        Grayscale image with values in [0, 1], shape (M,N).
    thres : float
        Positive threshold applied to the gradient magnitude.

    Returns
    -------
    out_img_array : np.ndarray (int)
        Binary edge map, same shape (M,N), values {0, 1}.
    """
    if in_img_array.ndim != 2:
        raise ValueError("in_img_array must be 2D")
    if thres <= 0:
        raise ValueError("thres must be positive")

    # ---------- Sobel kernels ----------
    Gx = np.array(
        [[-1, 0, +1],
         [-2, 0, +2],
         [-1, 0, +1]],
        dtype=float,
    )
    Gy = np.array(
        [[+1, +2, +1],
         [ 0,  0,  0],
         [-1, -2, -1]],
        dtype=float,
    )

    # The centre of the 3×3 mask is (1, 1)
    mask_origin = np.array([1, 1], dtype=int)
    in_origin = np.array([0, 0], dtype=int)

    # ---------- convolve with Sobel X ----------
    conv_x, origin_x = fir_conv(in_img_array, Gx, in_origin, mask_origin)
    # crop back to original size using the reported origin
    start_r, start_c = origin_x
    end_r = start_r + in_img_array.shape[0]
    end_c = start_c + in_img_array.shape[1]
    gx = conv_x[start_r:end_r, start_c:end_c]

    # ---------- convolve with Sobel Y ----------
    conv_y, origin_y = fir_conv(in_img_array, Gy, in_origin, mask_origin)
    start_r, start_c = origin_y
    end_r = start_r + in_img_array.shape[0]
    end_c = start_c + in_img_array.shape[1]
    gy = conv_y[start_r:end_r, start_c:end_c]

    # ---------- gradient magnitude ----------
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # ---------- threshold ----------
    out_img_array = (grad_mag >= thres).astype(int)

    return out_img_array
