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


def apply_color_range_to_image(original_image, color_range):
    binary_mask = np.full_like(original_image[:, :, 0], True)
    binary_mask = np.where(original_image[:, :, 2] < color_range[0][1], binary_mask, False)
    binary_mask = np.where(original_image[:, :, 2] > color_range[0][0], binary_mask, False)
    binary_mask = np.where(original_image[:, :, 1] < color_range[1][1], binary_mask, False)
    binary_mask = np.where(original_image[:, :, 1] > color_range[1][0], binary_mask, False)
    binary_mask = np.where(original_image[:, :, 0] < color_range[2][1], binary_mask, False)
    binary_mask = np.where(original_image[:, :, 0] > color_range[2][0], binary_mask, False)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    binary_mask = np.where(map_hue(hsv_image[:, :, 0]) < color_range[3][1], binary_mask, False)
    binary_mask = np.where(map_hue(hsv_image[:, :, 0]) > color_range[3][0], binary_mask, False)
    binary_mask = np.where(hsv_image[:, :, 1] < color_range[4][1], binary_mask, False)
    binary_mask = np.where(hsv_image[:, :, 1] > color_range[4][0], binary_mask, False)
    return binary_mask

def get_masked_image(image, binary_mask):
    return np.where(np.stack([binary_mask]*3, -1), image, np.zeros_like(image))


def part1(mask_images):
    binary_masks = []
    for mask_img in mask_images:
        binary_mask = np.logical_or(mask_img[:, :, 0] > 0,
                                               np.logical_or(mask_img[:, :, 1] > 0, mask_img[:, :, 2] > 0))
        binary_masks.append(binary_mask)
    return binary_masks


def part2(original_images, masked_images, binary_skin_pixel_masks):
    ranges = []
    for mask, binary_mask in zip(masked_images, binary_skin_pixel_masks):
        ranges.append(get_color_range_values(mask, binary_mask))

    binary_masks = []
    for img, rng in zip(original_images, ranges):
        binary_masks.append(apply_color_range_to_image(img, rng))

    masked_images = []
    for img, binary_mask in zip(original_images, binary_masks):
        masked_images.append(get_masked_image(img, binary_mask))

    for i, img in enumerate(masked_images):
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_FOLDER, 'img_%03d_masked.jpg' % (i+1) ), img)


# Using morpological operations
def part3(original_images, binary_masks):
    for i, (img, mask) in enumerate(zip(original_images, binary_masks)):
        eroded = cv2.erode(mask.astype(np.float), np.ones((11, 11), dtype=float), iterations=1).astype(np.bool)
        color_range = get_color_range_values(img, eroded)
        binary_mask = apply_color_range_to_image(img, color_range)
        masked = get_masked_image(img, binary_mask)
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_FOLDER, 'img_%03d_eroded.jpg' % (i + 1)), masked)

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


binary_masks = part1(mask_images)
part2(orig_images, mask_images, binary_masks)
part3(orig_images, binary_masks)