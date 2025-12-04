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

test_path = "./dataset/test"
for img_name in os.listdir(test_path):
    img_path = f"{test_path}/{img_name}"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(img_gray)

    if len(detected_faces) < 1:
        continue

    for face in detected_faces:
        x, y, h, w = face
        face_img = img_gray[y : y + h, x : x + w]

        label, confidence = recognizer.predict(face_img)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{person_names[label]}: {confidence:.2f}"
        cv2.putText(
            img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("Result", img)

    cv2.waitKey(0)
cv2.destroyAllWindows()
