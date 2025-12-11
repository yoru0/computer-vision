import os
import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

train_dir = "images/train/"
test_dir = "images/test/"

classes = os.listdir(train_dir)


# training data
def train_test_model():
    face_list = []
    class_id_list = []

    for class_id, parent_path in enumerate(classes):
        face_dir = os.path.join(train_dir, parent_path)
        face_paths = os.listdir(face_dir)

        for path in face_paths:
            full_path = os.path.join(face_dir, path)
            gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(detected_faces) < 1:
                continue

            for rect in detected_faces:
                x, y, w, h = rect
                face_img = gray[y : y + h, x : x + w]
                face_list.append(face_img)
                class_id_list.append(class_id)

    # train
    face_recognizer.train(face_list, np.array(class_id_list))

    print("Training completed.")

    # testing
    total = 0
    correct = 0

    for class_id, parent_path in enumerate(classes):
        test_actor_dir = os.path.join(test_dir, parent_path)

        for filename in os.listdir(test_actor_dir):
            full_face_path = os.path.join(test_actor_dir, filename)
            img = cv2.imread(full_face_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(detected_faces) < 1:
                continue

            for x, y, w, h in detected_faces:
                face_img = gray[y : y + h, x : x + w]
                predict_label, _ = face_recognizer.predict(face_img)

                total += 1

                if predict_label == class_id:
                    correct += 1

    accuracy = correct / total * 100
    print(f"Prediction accuracy: {accuracy:.2f}%")

    face_recognizer.write("lbph_model.xml")


def predict_picture():
    if not os.path.exists("lbph_model.xml"):
        print("Model doesn't exist.")
        return

    face_recognizer.read("lbph_model.xml")
    path_input = input("Enter image path: ")

    if not os.path.exists(path_input):
        print("Image path doesn't exist.")
        return

    img = cv2.imread(path_input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(detected_faces) < 1:
        print("No faces detected.")
        return

    for x, y, w, h in detected_faces:
        face_img = gray[y : y + h, x : x + w]
        res, conf = face_recognizer.predict(face_img)

        conf = math.floor(conf * 100) / 100
        name = classes[res]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = name + " " + str(conf)

        cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu():
    while True:
        print("1. Train")
        print("2. Predict")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_test_model()
        elif choice == "2":
            predict_picture()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

menu()