from pathlib import Path
import numpy as np
from PIL import Image
from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough
import cv2

def load_gray(path: Path) -> np.ndarray:
    """Open image with Pillow, return float32 array in [0,1]."""
    img = Image.open(path).convert("L")      # convert to 8‑bit greyscale
    return np.asarray(img, dtype=np.float32) / 255.0


def save_png(path: Path, img: np.ndarray) -> None:
    """Save array as PNG"""
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# def resize_custom_func(sobel_edges: np.ndarray) -> np.ndarray:
#     h, w = sobel_edges.shape[:2]
#     resized = resize(
#         sobel_edges,
#         (h // 4, w // 4),
#         anti_aliasing=False,
#         preserve_range=True
#     ).astype(np.uint8)
#     return resized

def resize_custom_func(string_img: str) -> None:
    img = iio.imread(string_img + ".png")
    h, w = img.shape[:2]
    img_small = resize(
        img,
        (h // 4, w // 4),
        anti_aliasing=True,
        preserve_range=True
    ).astype(img.dtype)
    iio.imwrite(Path(f"{string_img}_small.png"), img_small)


def main() -> None:
    img_path = Path("basketball_large.png")  # RGB test image
    sobel_thres = 0.25                      # gradient threshold
    V_min = 40                          # minimum votes for Hough
    R_max = 200                        # maximum radius for Hough
    nx, ny, nr = 50, 50, 30                 # Hough bins
    
    if not img_path.exists():
        raise FileNotFoundError(f"Input image {img_path} not found")

    # ---- load and preprocessing ----
    img_gray = load_gray(img_path)
    print(f"Loaded {img_path.name}, shape {img_gray.shape}")

    # ---- Sobel edges ----
    sobel_edges = sobel_edge(img_gray, sobel_thres)
    save_png(img_path.with_name("sobel.png"), sobel_edges * 255)
    print("Saved sobel.png")

    # ---- log edges ----
    log_edges = log_edge(img_gray)
    save_png(img_path.with_name("log.png"), log_edges * 255)
    print("Saved log.png")

    # new_sobel_edges = resize_custom_func(sobel_edges)
    # new_log_edges = resize_custom_func(log_edges)

    # print(sobel_edges.shape)
    # print(log_edges.shape)
    # print(new_sobel_edges.shape)
    # print(new_log_edges.shape)

    resize_custom_func("sobel")
    resize_custom_func("log")

    new_sobel_edges = cv2.imread("sobel_small.png")
    new_log_edges = cv2.imread("log_small.png") 

    # ---- Hough on Sobel edge map ----
    dim = np.array([nx, ny, nr], dtype=int)
    centers, radii = circ_hough(new_sobel_edges, R_max, dim, V_min, log=False)
    print("\nCircles from Sobel + Hough:")
    if radii.size == 0:
        print("  (none above vote threshold)")
    else:
        for (row, col), r in zip(centers, radii):
           print(f"  row={row:.1f}, col={col:.1f}, r={r:.1f}")

    # ---- Hough on log edge map ----
    centers2, radii2 = circ_hough(new_log_edges, R_max, dim, V_min, log=True)
    print("\nCircles from LoG + Hough:")
    if radii2.size == 0:
        print("  (none above vote threshold)")
    else:
        for (row, col), r in zip(centers2, radii2):
            print(f"  row={row:.1f}, col={col:.1f}, r={r:.1f}")


if __name__ == "__main__":
    main()
