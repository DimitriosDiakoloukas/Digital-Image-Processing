import cv2
import numpy as np
from typing import Tuple
from PIL import Image
from pathlib import Path

def find_hough_circles(image, edge_image, r_min, R_max, delta_r, num_thetas, bin_threshold, post_process=True, top_k=20):
    """
    Find circles in an image using the Hough transform.
    Parameters
    ----------
    image : np.ndarray
        Input image, either grayscale or RGB.
    edge_image : np.ndarray
        Edge-detected image (binary) using Canny or Sobel.
    r_min : float   
        Minimum radius of circles to detect.
    R_max : float
        Maximum radius of circles to detect.
    delta_r : float 
        Step size for radius.
    num_thetas : int
        Number of theta values to sample.
    bin_threshold : float   
        Minimum number of votes for a circle to be considered detected.
    post_process : bool 
        If True, post-process the detected circles to remove near-duplicates.
    top_k : int 
        Number of top circles to return based on vote strength.
    Returns 
    -------
    output_img : np.ndarray
        Image with detected circles drawn on it.
    all_circles : list of tuples
        List of detected circles, each represented as (x, y, r, votes).
    """
    img_height, img_width = edge_image.shape[:2]
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, R_max, step=delta_r)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = [(r, int(r * cos_thetas[t]), int(r * sin_thetas[t]))
                         for r in rs for t in range(num_thetas)]

    from collections import defaultdict
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y, x] != 0:
                for r, dx, dy in circle_candidates:
                    xc = x - dx
                    yc = y - dy
                    if 0 <= xc < img_width and 0 <= yc < img_height:
                        accumulator[(xc, yc, r)] += 1

    output_img = image.copy()
    all_circles = [
        (x, y, r, votes / num_thetas)
        for (x, y, r), votes in accumulator.items()
        if votes / num_thetas >= bin_threshold
    ]

    # Sort by vote strength
    all_circles.sort(key=lambda c: -c[3])
    if top_k:
        all_circles = all_circles[:top_k]

    # Post-process: remove near-duplicates
    if post_process:
        final_circles = []
        for x, y, r, v in all_circles:
            too_close = False
            for xc, yc, rc, _ in final_circles:
                if np.hypot(x - xc, y - yc) < 10 and abs(r - rc) < 5:
                    too_close = True
                    break
            if not too_close:
                final_circles.append((x, y, r, v))
        all_circles = final_circles

    for x, y, r, _ in all_circles:
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 1)  # thin stroke for circles

    return output_img, all_circles


def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int, log: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Detect circles in an image using Hough transform. This function is a wrapper for the find_hough_circles function, "
        taking care of the input image and parameters, and returning the centers and radii of the detected circles.
    Parameters 
    ----------
    in_img_array : np.ndarray
        Input image, either grayscale or RGB.
    R_max : float
        Maximum radius of circles to detect.
    dim : np.ndarray
        Dimensions of the Hough transform, containing [nx, ny, nr].
    V_min : int
        Minimum number of votes for a circle to be considered detected.
    log : bool
        If True, the input image is assumed to be a log edge image; otherwise, it is assumed to be a Sobel edge image.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        centers : np.ndarray
            Array of shape (N, 2) containing the (x, y) coordinates of the detected circle centers.
        radii : np.ndarray
            Array of shape (N,) containing the radii of the detected circles.
    """
    R_min = 95
    delta_r = 1
    num_thetas = 100
    bin_threshold = V_min / num_thetas  # I define Vmin as the minimum number of votes and is used to calculate the bin threshold 
                                        # which is the minimum number of votes required to consider a circle as detected and is 0.4 in my case.
    min_edge_threshold = 100
    max_edge_threshold = 200
    
    input_img = in_img_array.copy()

    if input_img.ndim == 3:
        edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        print("Converting to grayscale")
    else:
        edge_image = input_img.copy()
        print("Image is already grayscale")
    
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
    
    if edge_image is not None:
        
        print ("Detecting Hough Circles Started!")
        circle_img, circles = find_hough_circles(input_img, edge_image, R_min, R_max, delta_r, num_thetas, bin_threshold)
        
        if circle_img is not None:
            if not log:
                cv2.imwrite("circles_img_sobel.png", circle_img)
            else:
                cv2.imwrite("circles_img_log.png", circle_img)
    else:
        print ("Error in input image!")
            
    print("Detecting Hough Circles Complete!")

    centers = np.array([[x, y] for x, y, r, v in circles], dtype=float)
    radii   = np.array([r for x, y, r, v in circles], dtype=float)

    return centers, radii