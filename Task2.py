import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

IMAGE_FOLDER = os.path.join('Images', 'Ground Truths')

# OpenCV uses BGR instead og RGB, that's why channels are different with
# matplotlib


def display_image(img, bgr=True):
    plt.figure()
    if bgr:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()


filenames = [img for img in glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))]

for f in filenames:
	pass
	