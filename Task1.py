import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to original image
IMAGE_FILE = os.path.join('Images', 'Original Images', 'img_001.jpg')

# OpenCV uses BGR instead og RGB, that's why channels are different with matplotlib
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
        single_channel[:,:,channel_index] = img[:,:,channel_index]
        return single_channel
    return img[:,:,channel_index]

# Creates histogram from the specified channel of specified cimage
def create_histogram(img, channel):
    hist = np.zeros(256)
    for i in img[:,:,channel]:
        for j in i:
            hist[i] += 1
        
    return hist

# Hue values are between (0, 179) in OpenCV, it is mapped to (0, 255) for visualization purposes
def map_hue(img, max_hue_val=179):
     return (img * (max_hue_val / 255)).astype(int)
        
# Part 1 of the Task 1, create single R, G, B channels from RGB images 
def part1(img, display=False):
    img_r = get_single_channel(img, 2)
    img_g = get_single_channel(img, 1)
    img_b = get_single_channel(img, 0)

    cv2.imwrite('img_001_r.jpg', img_r)
    if display: display_image(img_r)
    cv2.imwrite('img_001_g.jpg', img_g)
    if display: display_image(img_g)
    cv2.imwrite('img_001_b.jpg', img_b)
    if display: display_image(img_b)
    return img

# Part 2 of the Task 1, create single H, S, V channels from HSV images 
def part2(img, display=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img_h = map_hue(get_single_channel(img_hsv, 0, rgb=False))
    img_s = get_single_channel(img_hsv, 1, rgb=False)
    img_v = get_single_channel(img_hsv, 2, rgb=False)

    img_hsv = np.stack((img_h, img_s, img_v), axis=2)

    cv2.imwrite('img_001_h.jpg', img_h)
    if display: display_image(img_h, bgr=False)
    cv2.imwrite('img_001_s.jpg', img_s)
    if display: display_image(img_s, bgr=False)
    cv2.imwrite('img_001_v.jpg', img_v)
    if display: display_image(img_v, bgr=False)
    return img_hsv

# Creating hist
def part3(img, img_hsv):
    hist_r = create_histogram(img, 2)
    hist_g = create_histogram(img, 1)
    hist_b = create_histogram(img, 0)

    hist_h = create_histogram(img_hsv, 0)
    hist_s = create_histogram(img_hsv, 1)
    hist_v = create_histogram(img_hsv, 2)



    return 


img = cv2.imread(IMAGE_FILE)

img = part1(img)
img_hsv = part2(img)
hist = part3(img, img_hsv)



