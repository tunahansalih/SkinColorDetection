import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#Path to original image
IMAGE_FILE = os.path.join('Images', 'Original Images', 'img_001.jpg')

# OpenCV uses BGR instead og RGB, that's why channels are different with matplotlib
def display_image(img, bgr=True):
    plt.figure()
    if bgr:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()
         
    
def get_single_channel(img, channel_index):
    single_channel = np.zeros_like(img)
    single_channel[:,:,channel_index] = img[:,:,channel_index]
    return single_channel

def create_histogram(img, channel):
    hist = np.zeros(255)
    for i in img[:,:,channel]:
        hist[i] += 1
        
    return hist

def map_hue(img, max_hue_val=179):
     return (img * (max_hue_val / 255)).astype(int)
        

def part1(img):
    img_r = get_single_channel(img, 2)
    img_g = get_single_channel(img, 1)
    img_b = get_single_channel(img, 0)

    cv2.imwrite('img_001_r.jpg', img_r)
    display_image(img_r)
    cv2.imwrite('img_001_g.jpg', img_g)
    display_image(img_g)
    cv2.imwrite('img_001_b.jpg', img_b)
    display_image(img_b)

def part2(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_h = map_hue(get_single_channel(img_hsv, 0))
    img_s = get_single_channel(img_hsv, 1)
    img_v = get_single_channel(img_hsv, 2)

    img_hsv = img_h + img_s + img_v
    cv2.imwrite('img_001_hsv.jpg', img_hsv)

    cv2.imwrite('img_001_h.jpg', img_h)
    display_image(img_h, bgr=False)
    cv2.imwrite('img_001_s.jpg', img_s)
    display_image(img_s, bgr=False)
    cv2.imwrite('img_001_v.jpg', img_v)
    display_image(img_v, bgr=False)

img = cv2.imread(IMAGE_FILE)

part1(img)
part2(img)
part3(img)



