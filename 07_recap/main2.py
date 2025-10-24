import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


data_folder = "Images/Data/"
data = []
for img_path in os.listdir(data_folder):
    img_path = data_folder + img_path
    img_data = cv2.imread(img_path)
    data.append(img_data)

images = [
    "Images/Object.png",
    "Images/Object2.png",
    "Images/Object3.png",
]

for object_path in images:
    object = cv2.imread(object_path)
    object = cv2.cvtColor(object, cv2.COLOR_BGR2RGB)
    gray_object = cv2.cvtColor(object, cv2.COLOR_RGB2GRAY)
    gray_object = cv2.medianBlur(gray_object, ksize=5)
    gray_object = cv2.equalizeHist(gray_object)

    akaze = cv2.AKAZE_create()
    target_kp, target_desc = akaze.detectAndCompute(gray_object, None)
    target_desc = np.float32(target_desc)

    best_match = 0
    for index, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_data = cv2.medianBlur(gray_data, ksize=5)
        gray_data = cv2.equalizeHist(gray_data)

        data_kp, data_desc = akaze.detectAndCompute(gray_data, None)
        data_desc = np.float32(data_desc)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))
        matcher = flann.knnMatch(target_desc, data_desc, 2)

        matchmask = [[0, 0] for _ in range(0, len(matcher))]

        curr_matches = 0
        for i, (fm, sm) in enumerate(matcher):
            if fm.distance < 0.7 * sm.distance:
                matchmask[i] = [1, 0]
                curr_matches += 1

        if best_match < curr_matches:
            best_match = curr_matches
            best_match_data = {
                "image_data": img,
                "keypoint": data_kp,
                "descriptor": data_desc,
                "match": matcher,
                "matchmask": matchmask,
            }


    result = cv2.drawMatchesKnn(
        object,
        target_kp,
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
