import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

ORIG_IMAGE_FOLDER = os.path.join('Images', 'Original Images')
MASK_IMAGE_FOLDER = os.path.join('Images', 'Ground Truths')
OUTPUT_IMAGE_FOLDER = os.path.join('Images', 'Output Images')
# OpenCV uses BGR instead og RGB, that's why channels are different with
# matplotlib

def create_bitmask(img):
    return np.where(np.equal(img, np.zeros_like(img)), npnp.full_like())

def display_image(img, bgr=True):
    plt.figure()
    if bgr:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()


orig_filenames = [img for img in glob.glob(os.path.join(ORIG_IMAGE_FOLDER, "*.jpg"))][:10]
mask_filenames = [img for img in glob.glob(os.path.join(MASK_IMAGE_FOLDER, "*.jpg"))][:10]
orig_images = []
mask_images = []

for f in orig_filenames:
    orig_images.append(cv2.imread(f))

for f in mask_filenames:
    mask_images.append(cv2.imread(f))

def part1(mask_images):
    binary_skin_pixel_masks = []
    for mask_img in mask_images:
        mask_img[mask_img > 0] = 255
        binary_skin_pixel_masks.append(mask_img)
    return binary_skin_pixel_masks


binary_skin_pixel_masks = part1(mask_images)


