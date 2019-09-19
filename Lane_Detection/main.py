import cv2
import sys
import os

sys.path.append("/Lane_Detection")
from lane_detection import lane_detection
#, get_curve, draw_lanes

img = cv2.imread('/home/hackathon/Curved-Lane-Lines/test_images/test3.jpg')
#plt.imshow(img)
size = (100,100)
lane_detection(img,size)
#curverad =get_curve(img, curves[0], curves[1])
#lane_curve = np.mean([curverad[0], curverad[1]])
#img_n = draw_lanes(img, curves[0], curves[1])
