import cv2
import os
import matplotlib.pyplot as plt

files = os.listdir("../dataset")

for i, f in enumerate(files):
    img = cv2.imread(os.path.join("../dataset", f))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    fast_res = img.copy()
    cv2.drawKeypoints(fast_res, keypoints, fast_res, (0, 0, 255))

    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    orb_res = img.copy()
    cv2.drawKeypoints(orb_res, keypoints, orb_res, (0, 255, 0))

    plt.subplot(len(files), 2, 2 * i + 1)
    plt.title(f"FAST: {f}")
    fast_res = cv2.cvtColor(fast_res, cv2.COLOR_BGR2RGB)
    plt.imshow(fast_res)

    plt.subplot(len(files), 2, 2 * i + 2)
    plt.title(f"ORB: {f}")
    orb_res = cv2.cvtColor(orb_res, cv2.COLOR_BGR2RGB)
    plt.imshow(orb_res)

plt.show()
