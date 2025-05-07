from pathlib import Path
import numpy as np
from PIL import Image

from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough


def load_gray(path: Path) -> np.ndarray:
    """Open image with Pillow, return float32 array in [0,1]."""
    img = Image.open(path).convert("L")      # convert to 8‑bit greyscale
    return np.asarray(img, dtype=np.float32) / 255.0


def save_png(path: Path, img: np.ndarray) -> None:
    """Save array as PNG"""
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main() -> None:
    img_path = Path("basketball_large.png")  # RGB test image
    sobel_thres = 0.25                      # gradient threshold
    r_max = 60.0                            # max circle radius
    nx, ny, nr = 50, 50, 30                 # Hough bins
    votes_min = 20                          # Hough vote cut‑off

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

    # ---- Hough on Sobel edge map ----
    dim = np.array([nx, ny, nr], dtype=int)
    centers, radii = circ_hough(sobel_edges, r_max, dim, votes_min)

    print("\nCircles from Sobel + Hough:")
    if radii.size == 0:
        print("  (none above vote threshold)")
    else:
        for (row, col), r in zip(centers, radii):
            print(f"  row={row:.1f}, col={col:.1f}, r={r:.1f}")

    # ---- Hough on log edge map ----
    centers2, radii2 = circ_hough(log_edges, r_max, dim, votes_min)
    print("\nCircles from LoG + Hough:")
    if radii2.size == 0:
        print("  (none above vote threshold)")
    else:
        for (row, col), r in zip(centers2, radii2):
            print(f"  row={row:.1f}, col={col:.1f}, r={r:.1f}")


if __name__ == "__main__":
    main()
