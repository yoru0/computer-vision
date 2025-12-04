import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Face Recognition (+ Training)
face_list = []
label_list = []

train_path = "./dataset/train"
person_names = os.listdir(train_path)

for idx, name in enumerate(person_names):
    dir_name = f"{train_path}/{name}"
    for file_name in os.listdir(dir_name):
        file_path = f"{dir_name}/{file_name}"
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        detected_faces = face_cascade.detectMultiScale(
            img, scaleFactor=1.2, minNeighbors=5
        )
        if len(detected_faces) < 1:
            continue

        for face in detected_faces:
            x, y, h, w = face
            face_img = img[y : y + h, x : x + w]

            face_list.append(face_img)
            label_list.append(idx)


# Face Detection (Testing)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_list, np.array(label_list))