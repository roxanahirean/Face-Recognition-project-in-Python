# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:22:12 2022

@author: Roxi
"""

import cv2
import face_recognition

#step1: get the default webcam video
#get the webcam #0 (the default one, 1, 2 etc means additional attached cams)
#capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

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
        #step9: draw rectangle around each face location in the main video frame inside the loop
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
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
