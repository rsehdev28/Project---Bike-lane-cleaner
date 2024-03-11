import cv2 
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)#changes the image into a grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0) # applies gaussian blur and removes noise and extra stuff sto prevent problems with edge detection
    canny = cv2.Canny(blur,50,150) # Detection of Sharp edges/ strong gradient of pixels in the image
    return canny

def region_of_interest(image):  # Identifies the region of interest by isolating the edge lines based on coordinates and using a mask
    height = image.shape[0]
    polygons = np.array([[(475,height),(600,height),(500,75)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
image = cv2.imread('test_image4.jpg') # Reads the image in the folder and returns it as multidimensional arrays containing intensities of each pixel. Try playing with by changing images.
lane_image = np.copy(image) #copies the image array into another variable for greyscaling
canny = canny(lane_image)
#cv2.imshow("Result",canny) renders the image and shows it with a window name
#cv2.waitKey(0)  Displays the image
#plt.imshow(canny) Displays the image with a graph having x and y axes 
#plt.show() 
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180) # To find the lines that best define our edges
cv2.imshow("result",cropped_image)
cv2.waitKey(0)
