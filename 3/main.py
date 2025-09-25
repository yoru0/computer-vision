import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMG_PATH = Path(__file__).resolve().parent.parent / "dataset" / "odi.jpg"

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image located at {IMG_PATH}")

# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
_, img_binary_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
_, img_tozero = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
_, img_tozero_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
_, img_trunc = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
_, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

thresh_result = [
    ("binary", img_binary),
    ("binary inversed", img_binary_inv),
    ("tozero", img_tozero),
    ("tozero inversed", img_tozero_inv),
    ("truncated", img_trunc),
    ("otsu", img_otsu),
]

plt.figure("Threshold Result", (6, 4))

for i, (title, img) in enumerate(thresh_result):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mean Filter
img_blur = cv2.blur(img, (5, 5))
# Median Filter
img_median = cv2.medianBlur(img, 5)
# Gaussian Blur
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
# Bilateral Filter
img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)

blur_result = [
    ("mean", img_blur),
    ("median", img_median),
    ("gaussian", img_gaussian),
    ("bilateral", img_bilateral),
]

plt.figure("Blur Result", (6, 4))

for i, (title, img) in enumerate(blur_result):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(img)
    plt.axis("off")

plt.tight_layout()
plt.show()
