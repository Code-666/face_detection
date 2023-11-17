# Import OpenCV package
import cv2

# Read image
image_path = "fd-test.jpg"
img = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check dimensions
print(img.shape)
print(gray_image.shape)

# Load the Classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Drawing a bounding box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Display the image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()
