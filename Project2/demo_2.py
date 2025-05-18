from pathlib import Path
import imageio.v3 as iio
from skimage.transform import resize

def resize_sobel():
    img = iio.imread("sobel.png")
    h, w = img.shape[:2]
    img_small = resize(
        img,
        (h // 4, w // 4),
        anti_aliasing=True,
        preserve_range=True
    ).astype(img.dtype)
    iio.imwrite(Path("sobel_small.png"), img_small)

if __name__ == "__main__":
    resize_sobel()
