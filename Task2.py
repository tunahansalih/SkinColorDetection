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


def get_nan_min_max(mask_image, binary_mask):
    mask_w_nan = np.where(binary_mask, mask_image, np.nan)
    return np.nanmin(mask_w_nan), np.nanmax(mask_w_nan)


def get_color_range_values(mask_image, binary_mask):
    r_range = get_nan_min_max(mask_image[:, :, 2], binary_mask)
    g_range = get_nan_min_max(mask_image[:, :, 1], binary_mask)
    b_range = get_nan_min_max(mask_image[:, :, 0], binary_mask)

    mask_image_hsv = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
    h_range = get_nan_min_max(map_hue(mask_image_hsv[:, :, 0]), binary_mask)
    s_range = get_nan_min_max(mask_image_hsv[:, :, 1], binary_mask)

    return r_range, g_range, b_range, h_range, s_range


def part1(mask_images):
    binary_skin_pixel_masks = []
    for mask_img in mask_images:
        binary_skin_pixel_mask = np.logical_or(mask_img[:, :, 0] > 0,
                                               np.logical_or(mask_img[:, :, 1] > 0, mask_img[:, :, 2] > 0))
        binary_skin_pixel_masks.append(binary_skin_pixel_mask)
    return binary_skin_pixel_masks


def part2(original_images, mask_images, binary_skin_pixel_masks):
    ranges = []
    for mask, binary_mask in zip(mask_images, binary_skin_pixel_masks):
        ranges.append(get_color_range_values(mask, binary_mask))

    print(ranges)



orig_filenames = [img for img in glob.glob(os.path.join(ORIG_IMAGE_FOLDER, "*.jpg"))][:10]
mask_filenames = [img for img in glob.glob(os.path.join(MASK_IMAGE_FOLDER, "*.jpg"))][:10]
orig_images = []
mask_images = []

orig_filenames.sort()
mask_filenames.sort()

for f in orig_filenames:
    orig_images.append(cv2.imread(f))

for f in mask_filenames:
    mask_images.append(cv2.imread(f))



binary_skin_pixel_masks = part1(mask_images)
part2(orig_images, mask_images, binary_skin_pixel_masks)
