# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:42:57 2022

@author: Roxi
"""


import cv2
import numpy as np
import face_recognition
import keras.utils as image
from keras.models import model_from_json

#step1: get the default webcam video
#get the webcam #0 (the default one, 1, 2 etc means additional attached cams)
#capture the video from default camera
webcam_video_stream = cv2.VideoCapture("C:/FAQULTATE/SSC LAB/Proiect Face Recognition/modi.mp4")

#face expression model initialization    
face_exp_model = model_from_json(open("C:/FAQULTATE/SSC LAB/Proiect Face Recognition/dataset/facial_expression_model_structure.json", "r").read())
#load weights into model
face_exp_model.load_weights('C:/FAQULTATE/SSC LAB/Proiect Face Recognition/dataset/facial_expression_model_weights.h5')
#list of emotions labels 
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


#step2: initialize empty array for face locations
#initialize the array variable to hold all the face locations in the frame
all_face_locations = []

#step3: create an outer while loop to loop through each frame of video
while True:
    #step4: get single frame of video as image
    #get current frame, ret is a boolean if the raid was successful
    ret, current_frame = webcam_video_stream.read()
    #step5: resize the frame to a quarter of size so that the computer can process it faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx = 0.25, fy = 0.25)
    #step6: find the total number of faces
    #find all face locations using face_locations() functions
    #model can be 'cnn' or 'hog'
    #number_of_times_to_upsample = 1 higher and detect more faces
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample = 2, model = 'hog')
    
    #step7: loop thorugh faces
    #looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        #split the tuple
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        #step8: 
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4   # we do * 4 to match the current frame, not the small one
        #printing the location of current face
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
    
        
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]        
    
        #step9: draw rectangle around each face location in the main video frame inside the loop
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
        
        
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
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
        
    #step11: display the current frame outside for loop inside the while loop
    cv2.imshow("Webcam Video ", current_frame)
    #step12: wait for a key press to breakthe while loop
    #press 'q' on the keyboard to break the while loop!
    #it remains 4bits
    if cv2.waitKey(1) & 0xFF == ord('q'):   #waitKey returns a 32b value of the key and with 0xFF we convert all the 28bits into zeros
        break
 
#step13: once loop breaks,release the camera resources and close all open windows
#release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
