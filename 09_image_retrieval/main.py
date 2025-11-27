import os
import cv2
from scipy.spatial.distance import euclidean

image_dir = 'assets/assets/image_library'
features = []

for filename in os.listdir(image_dir):
    img_name = filename.split('.')[0]
    img_bgr = cv2.imread(img_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    normalized = cv2.normalize(hist, None)

    flat_hist = normalized.flatten()
    features.append((img_name, flat_hist))