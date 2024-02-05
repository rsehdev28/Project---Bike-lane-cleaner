import cv2 
import numpy as np

image = cv2.imread('test_image1.jpg') # Reads the image in the folder and returns it as multidimensional arrays containing intensities of each pixel. Try playing with by changing images.
lane_image = np.copy(image) #copies the image array into another variable for greyscaling
gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
#cv2.imshow('result',gray) # renders the image and shows it wit a window name
#cv2.waitKey(0) # Displays the image

blur = cv2.GaussianBlur(gray,(5,5),0) # applies gaussian blur and removes noise and extra stuff sto prevent problems with edge detection
canny = cv2.Canny(blur,50,150) # Detection of Sharp edges/ strong gradient of pixels in the image
cv2.imshow('result',canny)
cv2.waitKey(0)