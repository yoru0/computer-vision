import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../dataset/odi.jpg")


def show_img(title, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(img)
    plt.show()


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)
harris = cv2.cornerHarris(img_gray, 2, 5, 0.01)

img_res = img.copy()
img_res[harris > 0.01 * harris.max()] = [0, 0, 255]

_, thresh = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
thresh = np.uint8(thresh)
_, _, _, centroid = cv2.connectedComponentsWithStats(thresh)
centroid = np.float32(centroid)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

subpix = cv2.cornerSubPix(img_gray, centroid, (5, 5), (-1, -1), criteria)
subpix = np.uint16(subpix)

subpix_res = img.copy()

for s in subpix:
    [x, y] = s
    subpix_res[y, x] = [0, 0, 255]