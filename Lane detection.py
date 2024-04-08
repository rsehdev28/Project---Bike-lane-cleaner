import cv2 
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)#changes the image into a grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0) # applies gaussian blur and removes noise and extra stuff sto prevent problems with edge detection
    canny = cv2.Canny(blur,50,150) # Detection of Sharp edges/ strong gradient of pixels in the image
    return canny

def make_coordinates(image, line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def display_lines(image,lines):
    line_image = np.zeros_like(image) #a zero value array containing values similar to the given image (all black)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            #All the lines that we detected in our graident image, we just drew them over a black image(line_image) with the same dimensions as out actual image
    return line_image

def average_slope_intercept(image,lines):
    left_fit =[]
    right_fit =[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis =0)
    right_fit_average =np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


def region_of_interest(image):  # Identifies the region of interest by isolating the edge lines based on coordinates and using a mask
    height = image.shape[0]
    polygons = np.array([[(475,height),(600,height),(500,75)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread('test_image.jpg') # Reads the image in the folder and returns it as multidimensional arrays containing intensities of each pixel. Try playing with by changing images.
lane_image = np.copy(image) #copies the image array into another variable for greyscaling
canny_image = canny(lane_image)

#cv2.imshow("Result",canny) #renders the image and shows it with a window name
#cv2.waitKey(0)  #Displays the image
#plt.imshow(canny_image) #Displays the image with a graph having x and y axes 
#plt.show() 

'''cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) # To find the lines that best define our edges
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,averaged_lines)
combo_image = cv2.addWeighted(lane_image,0.6,line_image,2,2)'''

#cv2.imshow("result",line_image)
#cv2.waitKey(0)
#cv2.imshow("result",combo_image)
#cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4") # Object to capture the video
while(cap.isOpened()):
    _, frame = cap.read() # To decode every video frame
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) # To find the lines that best define our edges
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame,0.6,line_image,2,2)
    cv2.imshow("result",combo_image)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

cv2.release()
cv2.destroyAllWindows()