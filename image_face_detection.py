# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:47:54 2022

@author: Roxi
"""

#import required libraries
import cv2
import face_recognition

#loading the image to detect  
image_to_detect = cv2.imread('C:/FAQULTATE/SSC LAB/Proiect Face Recognition/img_faces.jpg')

cv2.imshow("test", image_to_detect)


#find all face locations using face_locations() function
#model can be 'cnn' or 'hog'
# number_of_times_to_unsample = 1 higher and detect more faces

#detect all faces in the image
all_face_locations = face_recognition.face_locations(image_to_detect, model = 'hog')


#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))
 

#looping through the face locations
for index, current_face_location in enumerate(all_face_locations):
    #split the tuple
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
    #slice image array by positions inside the loop
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("Face No: " + str(index + 1), current_face_image)
    #show each sliced face inside the loop
    