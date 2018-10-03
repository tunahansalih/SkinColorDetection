import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_FILE = '/Users/tunahansalih/School/ComputerVision/SkinColorDetection/Images/Original Images/img_001.jpg'

img = cv2.imread(IMAGE_FILE)

# OpenCV uses BGR instead og RGB, that's why channels are different with matplotlib
def display_image(img):
    plt.figure()
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
        
    
img_r = get_single_channel(img, 2)
img_g = get_single_channel(img, 1)
img_b = get_single_channel(img, 0)

cv2.imwrite('img_001_r.jpg', img_r)
cv2.imwrite('img_001_g.jpg', img_g)
cv2.imwrite('img_001_b.jpg', img_b)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_h = map_hue(get_single_channel(img_hsv, 0))
cv2.imwrite('img_001_h.jpg', img_h)
img_s = get_single_channel(img_hsv, 1)
cv2.imwrite('img_001_s.jpg', img_s)
img_v = get_single_channel(img_hsv, 2)
cv2.imwrite('img_001_v.jpg', img_v)

