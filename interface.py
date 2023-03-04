# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 23:09:03 2022

@author: Roxi
"""


#import os
import subprocess
import cv2
import face_recognition
import numpy as np
import keras.utils as image
from keras.models import model_from_json

def run_file(key):
    if key == "1":
       #loading the image to detect  
       image_to_detect = cv2.imread('C:/FAQULTATE/SSC LAB/Proiect Face Recognition/img_faces.jpg')

       cv2.imshow("test", image_to_detect)
       
       cv2.waitKey(5000)


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
           cv2.waitKey(1000)
    elif key == "2":
        file_name = "realtime_face_detection.py"
        subprocess.run(["python", file_name])
    elif key == "3":
        file_name = "video_face_detection.py"
        subprocess.run(["python", file_name])
            
    elif key == "4":
        file_name = "realtime_face_detection_blur.py"
        subprocess.run(["python", file_name])
    
    elif key == "5":
        file_name = "image_face_emotion_detection.py"
        cv2.waitKey(1000)
        subprocess.run(["python", file_name])
        cv2.waitKey(10000)
            
    elif key == "6":
        file_name = "realtime_face_emotion_detection.py"
        subprocess.run(["python", file_name])
            
    elif key == "7":
            file_name = "video_face_emotion_detection.py"
            subprocess.run(["python", file_name])
    
# Wait for the user to press a key
key = input("Press a key: ")

# Run the face detection code
run_file(key)


   
        
    
        
   
        
 
        
    
        
    








