import cv2
import sys
import os
import time
from draw_lanes import draw_lanes
from get_curve import get_curve
sys.path.append("/Lane_Detection")
from lane_detection import lane_detection
#, get_curve, draw_lanes
import matplotlib.pyplot as plt
img = cv2.imread('/home/hackathon/Curved-Lane-Lines/test_images/test3.jpg')
#plt.imshow(img)
size = (100,100)
curves, lanes, ploty, out_img= lane_detection(img,size)
#curverad =get_curve(out_img, curves[0], curves[2])
#lane_curve = np.mean([curverad[0], curverad[1]])
startx = time.time()
img_resized = cv2.resize(img,(100,100))
img_n = draw_lanes(img_resized, curves[0], curves[2])
print("\n draw_lanes time is : "+ str(time.time()-startx))
f, (ax1) = plt.subplots(1, 1, figsize=(20,10))
ax1.imshow(img_n)
ax1.set_title('Original Image', fontsize=30)
