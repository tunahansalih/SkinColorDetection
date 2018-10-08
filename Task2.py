import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

ORIG_IMAGE_FOLDER = os.path.join('Images', 'Original Images')
MASK_IMAGE_FOLDER = os.path.join('Images', 'Ground Truths')
OUTPUT_IMAGE_FOLDER = os.path.join('Images', 'Output Images')


def display_image(img, bgr=True):
    plt.figure()
    if bgr:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()


def map_hue(img, max_hue_val=179):
    return (img * (255 / max_hue_val)).astype(int)


def get_nan_min_max(original_image, skin_pixel_mask_image):
    mask_w_nan = np.where(skin_pixel_mask_image, original_image[:, :, 2], np.nan)
    return np.nanmin(mask_w_nan), np.nanmax(mask_w_nan)


def get_color_range_values(original_image, skin_pixel_mask_image):
    r_range = get_nan_min_max(original_image[:, :, 2], skin_pixel_mask_image)
    g_range = get_nan_min_max(original_image[:, :, 1], skin_pixel_mask_image)
    b_range = get_nan_min_max(original_image[:, :, 0], skin_pixel_mask_image)

    original_image_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    h_range = get_nan_min_max(map_hue(original_image_hsv[:, :, 0]), skin_pixel_mask_image)
    s_range = get_nan_min_max(original_image_hsv[:, :, 1], skin_pixel_mask_image)

    return r_range, g_range, b_range, h_range, s_range


def part1(mask_images):
    binary_skin_pixel_masks = []
    for mask_img in mask_images:
        binary_skin_pixel_mask = np.logical_or(mask_img[:, :, 0] > 0,
                                               np.logical_or(mask_img[:, :, 1] > 0, mask_img[:, :, 2] > 0))
        binary_skin_pixel_masks.append(binary_skin_pixel_mask)
    return binary_skin_pixel_masks


def part2(original_images, binary_skin_pixel_masks):
    for original, mask in zip(original_images, binary_skin_pixel_masks):
        r = get_color_range_values(original, mask)


orig_filenames = [img for img in glob.glob(os.path.join(ORIG_IMAGE_FOLDER, "*.jpg"))][:10]
mask_filenames = [img for img in glob.glob(os.path.join(MASK_IMAGE_FOLDER, "*.jpg"))][:10]
orig_images = []
mask_images = []

for f in orig_filenames:
    orig_images.append(cv2.imread(f))

for f in mask_filenames:
    mask_images.append(cv2.imread(f))


binary_skin_pixel_masks = part1(mask_images)
part2(orig_images, binary_skin_pixel_masks)
