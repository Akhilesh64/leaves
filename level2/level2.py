#Importing required libraries
import numpy as np
import cv2
from PIL import Image
from skimage.measure import compare_ssim as ssim #for calculating similarity between features

#Converting to grayscale for resizing to compare similarity score 
maple = Image.open('mapleleafcorrect.jpg').convert('L')
neem = Image.open('neemleafcorrect.jpg').convert('L')

#Reading the images
img1 = cv2.imread('mapleleaves.jpg')
gray1 = cv2.imread('mapleleaves.jpg', 0)

img2 = cv2.imread('neemleaves.jpg')
gray2 = cv2.imread('neemleaves.jpg', 0)

'''
Thresholding the pics for contour detection and constructing bounding boxes.
The bounding box are required for cropping leaves present in the image and their
subsequent similarity prediction.
'''
_, thresh1 = cv2.threshold(gray1, 237, 255, 0)
thresh2 = cv2.adaptiveThreshold(gray2 , 255 ,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  #Adaptive thresholding yields better results than normal thresholding

contours1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
font = cv2.FONT_HERSHEY_COMPLEX  #setting the font for labelling

#Creating a function to calculate the leaf similarity 
def leaf_type(img, gray, cords):
    x, y, w, h = cords
    crop = gray[y:y+h, x:x+h]   #Cropping the image based on bounding boxes
    img = img.resize((crop.shape[1],crop.shape[0]))     #Resizing the original grayscale image to cropped image for calculating similarity
    img = np.array(img)
    (score, diff) = ssim(crop,img, full=True)           #Passing both the cropped images and resized image to calculate similarity
    return score

#Passing the contours for maple images for calculating similarity
for cnt in contours1: 
    x, y, w, h = cv2.boundingRect(cnt)
    if len(cnt) < 250:          #Setting a threshold to filter weaker predictions
        continue
    cords = (x, y, w, h)
    score = leaf_type(maple ,gray1, cords)      #Calling the leaf_type function which returns the calculated score 
    cv2.putText(img1, 'Maple : ' + str(round(score*100,2)), (x, y-3), font, 0.8, (255,0,0), 2) 

#Writing and visualizing the output
cv2.imshow('a',img1)
cv2.imwrite('output2_1.png',img1)

#Reapeating the same steps for neem leaves image
for cnt in contours2: 
    x, y, w, h = cv2.boundingRect(cnt)
    if len(cnt) < 345:
        continue
    cords = (x, y, w, h)
    score = leaf_type(neem, gray2, cords)
    cv2.putText(img2,'Neem : ' + str(round(score*100,2)), (x, y-3), font, 0.8, (255,0,0), 2) 

cv2.imshow('b',img2)
cv2.imwrite('output2_2.png',img2)
