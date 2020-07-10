#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os


# In[ ]:


KNOWN_FACES_DIR = "known_faces"

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"


a = r"C:\Users\LENOVO\Desktop\face\MuteVideo1594178332085[Trim][Merge].mp4"
cap = cv2.VideoCapture(a)
b = r"C:\Users\LENOVO\Desktop\face\cascade\data\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(b)

print("loading known faces")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        encoding = pickle.load(open(f"{name}/{filename}","rb"))
        known_faces.append(encoding)
        known_names.append(int(name))
if len(known_names)>0:
    next_id = max(known_names)+1
else:
    next_id = 0
print("processing unknown faces")

while True:
    ret, frame = cap.read()
    locations = face_recognition.face_locations(frame, model = MODEL)
    encodings = face_recognition.face_encodings(frame, locations)
    
    for face_encoding, face_location in zip(encodings,locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found:{match}")
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
            pickle.dump(face_encoding,open(f"{KNOWN_FACES_DIR}/{match}-{int(time.time())}.pkl","wb"))
            
            
        top_left = (face_location[3],face_location[0])
        bottom_right = (face_location[1],face_location[2])
        color = [0,255,0]
        cv2.rectangle(frame,top_left,bottom_right,color,FRAME_THICKNESS)
        
        top_left = (face_location[3],face_location[2])
        bottom_right = (face_location[1],face_location[2]+22)
        cv2.rectangle(frame,top_left,bottom_right,color,cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, match, (face_location[3]+10,face_location[2]+15),font ,0.5,(200,200,200))
        
    cv2.imshow("",frame)
    if cv2.waitKey(1) & 0xFF==("q"):
        break
        

