import cv2 
import numpy as np

image = cv2.imread('test_image.jpg') # Reads the image in the folder and returns it as multidimensional arrays containing intensities of each pixel.
lane_image = np.copy(image) #copies the image array into another variable for greyscaling
gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
cv2.imshow('result',gray) # renders the image and shows it wit a window name
cv2.waitKey(0) # Displays the image

