import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to original image
IMAGE_FILE = os.path.join('Images', 'Original Images', 'img_001.jpg')

# OpenCV uses BGR instead og RGB, that's why channels are different with
# matplotlib


def display_image(img, bgr=True):
    plt.figure()
    if bgr:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()

# Gets the specified channel from given image


def get_single_channel(img, channel_index, rgb=True):
    if rgb:
        single_channel = np.zeros_like(img)
        single_channel[:, :, channel_index] = img[:, :, channel_index]
        return single_channel
    return img[:, :, channel_index]

# Creates histogram from the specified channel of specified cimage


def create_histogram(img, channel, bin=256):
    hist = np.zeros(bin)
    size = 256 / bin
    for i in img[:, :, channel]:
        for j in i:
            hist[int(j / size)] += 1

    return hist


def display_histogram(hist, bin=256, xlabel='', fname='', disp=False):
    plt.figure()
    plt.bar(x=range(bin), height=hist, width=1.0)
    if xlabel != '':
        plt.xlabel(xlabel)
    plt.xlim(0, 255)
    plt.xticks(np.append(np.arange(0, 256, max(int(256 / bin), 32), 255)))
    plt.ylabel('# of Occurence')
    plt.box(on=None)
    if fname != '':
        plt.savefig(fname=fname)
    if disp:
        plt.show()
    pass

# Hue values are between (0, 179) in OpenCV, it is mapped to (0, 255) for
# visualization purposes


def map_hue(img, max_hue_val=179):
    return (img * (max_hue_val / 255)).astype(int)

# Part 1 of the Task 1, create single R, G, B channels from RGB images


def part1(img, display=False):
    img_r = get_single_channel(img, 2)
    img_g = get_single_channel(img, 1)
    img_b = get_single_channel(img, 0)

    cv2.imwrite('img_001_r.jpg', img_r)
    if display:
        display_image(img_r)
    cv2.imwrite('img_001_g.jpg', img_g)
    if display:
        display_image(img_g)
    cv2.imwrite('img_001_b.jpg', img_b)
    if display:
        display_image(img_b)
    return img

# Part 2 of the Task 1, create single H, S, V channels from HSV images


def part2(img, display=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_h = map_hue(get_single_channel(img_hsv, 0, rgb=False))
    img_s = get_single_channel(img_hsv, 1, rgb=False)
    img_v = get_single_channel(img_hsv, 2, rgb=False)

    img_hsv = np.stack((img_h, img_s, img_v), axis=2)

    cv2.imwrite('img_001_h.jpg', img_h)
    if display:
        display_image(img_h, bgr=False)
    cv2.imwrite('img_001_s.jpg', img_s)
    if display:
        display_image(img_s, bgr=False)
    cv2.imwrite('img_001_v.jpg', img_v)
    if display:
        display_image(img_v, bgr=False)
    return img_hsv

# Creating hist


def part3(img, img_hsv):
    hist_r = create_histogram(img, 2)
    display_histogram(hist_r, xlabel='Red Values', fname='hist_r')
    hist_g = create_histogram(img, 1)
    display_histogram(hist_g, xlabel='Green Values', fname='hist_g')
    hist_b = create_histogram(img, 0)
    display_histogram(hist_b, xlabel='Blue Values', fname='hist_b')

    hist_h = create_histogram(img_hsv, 0)
    display_histogram(hist_h, xlabel='Hue Values', fname='hist_h')
    hist_s = create_histogram(img_hsv, 1)
    display_histogram(hist_s, xlabel='Saturation Values', fname='hist_s')
    hist_v = create_histogram(img_hsv, 2)
    display_histogram(hist_v, xlabel='Brightness Values', fname='hist_v')


img = cv2.imread(IMAGE_FILE)

img = part1(img)
img_hsv = part2(img)
part3(img, img_hsv)
