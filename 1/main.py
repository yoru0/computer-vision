import cv2

image = cv2.imread("odi.jpg")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width, _ = image.shape
image[:, width//2:] = (0, 255, 255)

cv2.imwrite("modified_odi.jpg", image)