#!/usr/bin/env python
# coding: utf-8

# In[1]:

# importing the libraries
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


# In[ ]:

# uploading the video
a = r"C:\Users\LENOVO\Desktop\face\MuteVideo1594178332085[Trim][Merge].mp4"
cap = cv2.VideoCapture(a)
b = r"C:\Users\LENOVO\Desktop\face\cascade\data\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(b)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

