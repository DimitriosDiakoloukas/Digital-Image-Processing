from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def _to_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return arr / 255.0

def load_gray(path: Union[str, Path]) -> np.ndarray:
    img = Image.open(path).convert("L")  # 8‑bit greyscale
    return _to_array(img)


def load_rgb(path: Union[str, Path]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return _to_array(img)


if __name__ == "__main__":
    load_gray("basketball_large.png")
    load_rgb("basketball_large.png")