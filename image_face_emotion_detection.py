# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:51:15 2022

@author: Roxi
"""

import cv2
import numpy as np
import face_recognition
import keras.utils as image
from keras.models import model_from_json

#loading the image to detect  
image_to_detect = cv2.imread('C:/FAQULTATE/SSC LAB/Proiect Face Recognition/img_faces.jpg')

cv2.waitKey(5000)
#cv2.imshow("test", image_to_detect)

#face expression model initialization    
face_exp_model = model_from_json(open("C:/FAQULTATE/SSC LAB/Proiect Face Recognition/dataset/facial_expression_model_structure.json", "r").read())
#load weights into model
face_exp_model.load_weights('C:/FAQULTATE/SSC LAB/Proiect Face Recognition/dataset/facial_expression_model_weights.h5')
#list of emotions labels 
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


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
    #cv2.imshow("Face No: " + str(index + 1), current_face_image) 
    #show each sliced face inside the loop
    
    #step9: draw rectangle around each face location in the main video frame inside the loop
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
    
    
    #preprocessinput, convert it to an image like as the dataset
    #convert into grayscale
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    #resize to 48x48 px size
    current_face_image = cv2.resize(current_face_image, (48, 48))
    #convert the PIL image into a 3d numpy array
    img_pixels = image.img_to_array(current_face_image)
    #expand the shape of an array into a single row multiple columns
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    #pixels are in range of [0, 255], normalize all pixels in scale of [0, 1]
    img_pixels /= 255
    
    #do prediction using model, get the prediction values for all 7 expressions
    exp_predictions = face_exp_model.predict(img_pixels)
    #find max indexed prediction value (o till 7)
    max_index = np.argmax(exp_predictions[0])
    #get corresponding lable from emotions_label
    emotion_label = emotions_label[max_index]
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
#step11: display the current frame outside for loop inside the while loop
cv2.imshow("Image Face Emotions ", image_to_detect)
cv2.waitKey(5000)