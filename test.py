import face_recognition
import cv2
from openpyxl import Workbook
import datetime
import os
print(os.getcwd())
thisdir = os.getcwd()
known_face_encodings = []
known_face_names_txt = []
known_face_names = []

for r,d,f in os.walk(thisdir):
    counter = 1
    for file in f:
        if ".jpeg" in file:
            image_borhan = face_recognition.load_image_file(os.getcwd() + "/image/borhan.jpeg")
            image_borhan_face_encoding = face_recognition.face_encodings(image_borhan)[0]
            known_face_encodings.append(image_borhan_face_encoding)
            known_face_names_txt.append( os.path.splitext(file)[0])
            known_face_names.append(counter)
            print(counter,os.path.splitext(file)[0])
            counter+=1