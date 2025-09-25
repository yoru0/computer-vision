import cv2
from pathlib import Path

IMG_PATH = Path(__file__).resolve().parent.parent / "dataset" / "odi.jpg"

image = cv2.imread(str(IMG_PATH))
if image is None:
    raise FileNotFoundError(f"Could not load image located at {IMG_PATH}")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width, _ = image.shape
image[:, width // 2 :] = (0, 255, 255)

cv2.imwrite("modified_odi.jpg", image)
