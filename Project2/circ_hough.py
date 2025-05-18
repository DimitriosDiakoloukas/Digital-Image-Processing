import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple

def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process=True):
    img_height, img_width = edge_image.shape[:2]

    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, r_max, step=delta_r)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    accumulator = np.zeros((img_height, img_width, len(rs)), dtype=np.uint16)

    ys, xs = np.nonzero(edge_image)
    for x, y in zip(xs, ys):
        for r, rcos_t, rsin_t in circle_candidates:
            x_center = x - rcos_t
            y_center = y - rsin_t
            r_idx = int((r - r_min) / delta_r)
            if 0 <= x_center < img_width and 0 <= y_center < img_height:
                accumulator[y_center, x_center, r_idx] += 1

    output_img = image.copy()
    out_circles = []

    # NEW: Extract detected circles from dense accumulator
    y_idxs, x_idxs, r_idxs = np.nonzero(accumulator)
    for y, x, r_idx in zip(y_idxs, x_idxs, r_idxs):
        votes = accumulator[y, x, r_idx]
        vote_fraction = votes / num_thetas
        if vote_fraction >= bin_threshold:
            r = int(r_min + r_idx * delta_r)
            out_circles.append((x, y, r, vote_fraction))
            print(x, y, r, vote_fraction)

    # Post-process to remove duplicates
    if post_process:
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in sorted(out_circles, key=lambda t: -t[3]):
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold
                   for xc, yc, rc, _ in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    for x, y, r, v in out_circles:
        output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

    return output_img, out_circles


def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_min: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to use find_hough_circles with only 4 arguments as required by the assignment.
    Assumes:
      - r_min = 5
      - delta_r = 1
      - num_thetas = 100
      - bin_threshold = V_min / num_thetas
    """
    edge_image = in_img_array.astype(np.uint8) * 255
    input_img = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    r_min = 5
    delta_r = 1
    num_thetas = 100
    bin_threshold = V_min / num_thetas

    output_img, out_circles = find_hough_circles(
        input_img,
        edge_image,
        r_min,
        R_max,
        delta_r,
        num_thetas,
        bin_threshold,
        post_process=True
    )

    centers = np.array([[x, y] for x, y, r, v in out_circles], dtype=float)
    radii   = np.array([r for x, y, r, v in out_circles], dtype=float)

    return centers, radii
