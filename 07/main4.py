import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_folder = "Images/Data/"
data = [cv2.imread(data_folder + f) for f in os.listdir(data_folder)]

images = ["Images/Object.png", "Images/Object2.png", "Images/Object3.png"]

akaze = cv2.AKAZE_create()
flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))

for obj_path in images:
    obj = cv2.imread(obj_path)
    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    obj_gray = cv2.medianBlur(obj_gray, ksize=5)
    obj_gray = cv2.equalizeHist(obj_gray)
    obj_kp, obj_desc = akaze.detectAndCompute(obj_gray, None)

    best_match = 0
    for img in data:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, ksize=5)
        img_gray = cv2.equalizeHist(img_gray)
        img_kp, img_desc = akaze.detectAndCompute(img_gray, None)

        matches = flann.knnMatch(np.float32(obj_desc), np.float32(img_desc), 2)
        good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]

        if best_match < len(good):
            best_match = len(good)
            best_match_data = {"img":img, "kp":img_kp, "good": good}

    result = cv2.drawMatchesKnn(
        obj,
        obj_kp,
        best_match_data["img"],
        best_match_data["kp"],
        best_match_data["good"],
        None,
        matchColor=[255, 0, 0],
        singlePointColor=[0, 0, 255],
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Best match for {obj_path}")
    plt.show()
