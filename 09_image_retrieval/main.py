import os
import cv2
from scipy.spatial.distance import euclidean

image_dir = 'assets/assets/image_library'
features = []

for filename in os.listdir(image_dir):
    img_name = filename.split('.')[0]
    img_bgr = cv2.imread(image_dir + '/' + filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    normalized = cv2.normalize(hist, None)

    flat_hist = normalized.flatten()
    features.append((img_name, flat_hist))

test_image_bgr = cv2.imread('assets/assets/test_image/Geography.jpg')
test_image_rgb = cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2RGB)
test_hist = cv2.calcHist([test_image_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
test_normalized = cv2.normalize(test_hist, None)
test_flatten = test_normalized.flatten()

results = []
for name, hist in features:
    distance = euclidean(test_flatten, hist)
    results.append((distance, name))

sorted_results = sorted(results)
for distance, name in sorted_results:
    print(f'Image: {name}, Distance: {distance}')