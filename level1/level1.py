#Importing required libraries
import numpy as np
import cv2

#Reading the images in RGB and grayscale
img = cv2.imread('level1.jpg')
gray = cv2.imread('level1.jpg', 0)

#Performing thresholding for better contour detection 
ret, thresh = cv2.threshold(gray, 225, 255, 0)

#Finding the contours present in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_COMPLEX         #font for printing predictions

'''
Function to identify leaf type - 'old' or 'fresh'
Taking the original image and cordinates of bounding boxes to perform classification
'''
def leaf_type(img, cords):
    x, y, w, h = cords
    img_copy = img
    crop = img_copy[y:y+h, x:x+h]  #cropping the relevant part of the image containing the leaf
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) #Converting to HSV format for easier color detection

    #Specifying the lower and upper bounds of HSV values for color detection
    lower_green = np.array([40, 40, 40]) 
    upper_green = np.array([70, 255, 255])
    lower_yellow = np.array([20, 40, 40]) 
    upper_yellow = np.array([40, 255, 255])
    
    '''
    Creating a mask to detect the presence of color in the cropped image
    and summing the weights thus obtained for both yellow and green portions
    '''
    mask_green = cv2.inRange(crop, lower_green, upper_green)
    green = np.sum(mask_green)

    mask_yellow = cv2.inRange(crop, lower_yellow, upper_yellow)
    yellow = np.sum(mask_yellow)

    #Classifying the images on the basis of amount of color weight criteria provided
    if (yellow+green) != 0:     #To prevent divide by zero error
        ratio = green/(yellow+green)
        if ratio > 0.7:
            leaf = 'fresh'
            cv2.putText(img, leaf, (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
        elif ratio <= 0.7:
            leaf = 'old'
            cv2.putText(img, leaf, (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

'''
Looping through all the contours obtained to run the leaf_type function for
all the figures detected through contour detection as well as draw contour boundaries.
'''
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt)   #Getting the co-ordinates for bounding box
    if len(cnt)<20:    #Specifying a threshold for no of contours to filter wrong detections 
        continue
    cords = (x, y, w, h)
    leaf_type(img, cords)       #Calling leaf_type function for classification
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)  
    cv2.drawContours(img, contours, -1, (255,0,0), 2)  #draw boundary of contours.

#Saving and visualizing the output image
cv2.imshow('level1',img)
cv2.imwrite('output.png',img)





