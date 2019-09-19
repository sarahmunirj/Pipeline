from lane_detection import lane_detection

img = cv2.imread('~/home/hackathon/Curved-Lane-Lines/test_images/test3.jpg')
plt.imshow(img)
#size = (100,100)
lane_detection(img)