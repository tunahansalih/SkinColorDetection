import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_FILE = '/Users/tunahansalih/School/ComputerVision/SkinColorDetection/Images/Original Images/img_001.jpg'

img = cv2.imread(IMAGE_FILE, 1)

# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img[0] = np.zeros_like(img[0])

cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
