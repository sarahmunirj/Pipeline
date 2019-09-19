from pipeline import pipeline
from perspective_warp import perspective_warp
from inv_perspective import inv_perspective_warp
from sliding_window import sliding_window
from get_hist import get_hist
import time
import cv2

def lane_detection(img, size = (100,100)): 
  time_start = time.time()
  img = cv2.resize(img,size)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  time_1 = time.time()
  img_ = pipeline(img)
  time_2 = time.time()
  img_warp = perspective_warp(img_)
  time_3 = time.time()
  out_img, curves, lanes, ploty = sliding_window(img_warp, draw_windows=False)
  time_4 = time.time()
  
  print("\n time_color_channel_conv = " + str(int((time_1-time_start)*1000)) + "\n pipeline_time = "\
        + str(int((time_2-time_1)*1000)) + "\n perspective_warp_time = " \
        + str(int((time_3-time_2)*1000)) + "\n sliding_window_time = " + str(int((time_4-time_3)*1000)))
  print("\n Overall Time = " +str(int((time_4-time_start)*1000)))
              
  return curves, lanes, ploty, out_img
