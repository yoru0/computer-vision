import os
import cv2
import math
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

train_dir = "images/train"
test_dir = "images/test"

classes = os.listdir(train_dir)


def train_test_model():
    face_list = []
    class_id_list = []

    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)

        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            detected_face = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(detected_face) < 1:
                continue

            for x, y, w, h in detected_face:
                face_img = gray[y : y + h, x : x + w]
                face_list.append(face_img)
                class_id_list.append(class_id)

    face_recognizer.train(face_list, np.array(class_id_list))
    print("Training completed")

    total = 0
    correct = 0

    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)

        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            detected_face = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(detected_face) < 1:
                continue

            for x, y, w, h in detected_face:
                face_img = gray[y : y + h, x : x + w]
                predicted_id, _ = face_recognizer.predict(face_img)

                total += 1
                if predicted_id == class_id:
                    correct += 1

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")
    face_recognizer.write("model_v5.xml")


def predict_image():
    if not os.path.exists("model_v5.xml"):
        print("Model not found")
        return

    face_recognizer.read("model_v5.xml")
    image_path = input("Enter the absolute path to predict: ")

    if not os.path.exists(image_path):
        print("Image not found")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_face = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(detected_face) < 1:
        print("No face detected")
        return

    for x, y, w, h in detected_face:
        face_img = gray[y : y + h, x : x + w]
        predicted_id, confidence = face_recognizer.predict(face_img)

        confidence = math.floor(confidence * 100) / 100
        predicted_name = classes[predicted_id]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = predicted_name + " " + str(confidence)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    while True:
        print("1. Train and test model")
        print("2. Predict image")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_test_model()
        elif choice == "2":
            predict_image()
        elif choice == "3":
            break
        else:
            print("Invalid choice")


main()
