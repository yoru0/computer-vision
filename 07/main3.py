import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


PATH = "Images/"
object = cv2.imread(PATH + "Object.png")
# object = cv2.imread(PATH + 'Object2.png')

# Preprocessing
## Color to RGB
object = cv2.cvtColor(object, cv2.COLOR_BGRA2RGB)

# Akses dataset
DATA_PATH = PATH + "Data/"
data = []
for image_path in os.listdir(DATA_PATH):
    image_path = DATA_PATH + image_path
    image_data = cv2.imread(image_path)
    data.append(image_data)

# Preprocessing

## RGB to Grayscale
grayscale_object = cv2.cvtColor(object, cv2.COLOR_RGB2GRAY)

## Median Blur
grayscale_object = cv2.medianBlur(grayscale_object, ksize=5)

## Gausian Blur
# grayscale_object = cv2.GaussianBlur(grayscale_object, ksize=(3, 3), sigmaX=1)

## Equ Histogram
grayscale_object = cv2.equalizeHist(grayscale_object)


# Feature Descriptor

## AKAZE -> Accelerated KAZE
akaze = cv2.AKAZE_create()

target_keypoint, target_descriptor = akaze.detectAndCompute(grayscale_object, None)
target_descriptor = np.float32(target_descriptor)

# Best Matches
best_match = 0

for index, img in enumerate(data):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grayscale_data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # grayscale_data = cv2.medianBlur(grayscale_data, 5)
    grayscale_data = cv2.GaussianBlur(grayscale_data, (3, 3), 1)
    grayscale_data = cv2.equalizeHist(grayscale_data)

    data_keypoint, data_descriptor = akaze.detectAndCompute(grayscale_data, None)
    data_descriptor = np.float32(data_descriptor)

    # Match using FLANN
    flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))
    matcher = flann.knnMatch(target_descriptor, data_descriptor, 2)

    matchmask = [[0, 0] for _ in range(0, len(matcher))]

    current_matches = 0
    for i, (fm, sm) in enumerate(matcher):
        if fm.distance < 0.7 * sm.distance:
            matchmask[i] = [1, 0]  # mark as valid match
            current_matches += 1

    # Update best match
    if best_match < current_matches:
        best_match = current_matches
        best_match_data = {
            "image_data": img,
            "keypoint": data_keypoint,
            "descriptor": data_descriptor,
            "match": matcher,
            "matchmask": matchmask,
        }

# Plotting
result = cv2.drawMatchesKnn(
    object,
    target_keypoint,
    best_match_data["image_data"],
    best_match_data["keypoint"],
    best_match_data["match"],
    None,
    matchesMask=best_match_data["matchmask"],
    matchColor=[255, 0, 0],
    singlePointColor=[0, 0, 255],
)

plt.figure()
plt.imshow(result)
plt.title("Matching image")
plt.show()
