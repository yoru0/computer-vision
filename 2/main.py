import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("odi.jpg")

# Change image size
scale_percent = 1/2
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)

# Resize image
img = cv2.resize(img, dim, cv2.INTER_AREA)

# Convert to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display images
# cv2.imshow("Original", img)
# cv2.imshow("Grayscale", gray_image)
# cv2.waitKey(0)

h = img.shape[0]
w = img.shape[1]

intensity_counter = np.zeros(256, dtype=int)

for i in range(h):
    for j in range(w):
        intensity_counter[gray_image[i, j]] += 1

plt.figure(1)
plt.plot(intensity_counter, 'g', label='Odi')
plt.legend(loc='upper right')
plt.xlabel('Intensity')
plt.ylabel('Quantity')
# plt.show()

# Histogram equalization
equ = cv2.equalizeHist(gray_image)
equ_counter = np.zeros(256, dtype=int)
for i in range(h):
    for j in range(w):
        equ_counter[equ[i, j]] += 1

plt.figure(1, (16, 8))
plt.subplot(1, 2, 1)
plt.plot(intensity_counter, 'g', label='Before')
plt.legend(loc='upper right')
plt.xlabel('Intensity')
plt.ylabel('Quantity')

plt.subplot(1, 2, 2)
plt.plot(equ_counter, 'r', label='After')
plt.legend(loc='upper right')
plt.xlabel('Intensity')
plt.ylabel('Quantity')
# plt.show()

res = np.hstack((gray_image, equ))
cv2.imshow("Image", res)
cv2.waitKey(0)