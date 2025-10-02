import cv2
from pathlib import Path
from matplotlib import pyplot as plt

IMG_PATH = Path(__file__).resolve().parent.parent / "dataset" / "odi.jpg"

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image located at {IMG_PATH}")

# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow("Original", img)
# cv2.waitKey(0)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=5)

# plt.imshow(sobelx, "gray")
# plt.show()

# plt.imshow(sobely, "gray")
# plt.show()

# laplace
# laplace = cv2.Laplacian(grayscale, cv2.CV_64F)
# plt.imshow(laplace, "gray")
# plt.show()

# canny
canny = cv2.Canny(grayscale, 50, 200)
plt.imshow(canny, "gray")
plt.show()