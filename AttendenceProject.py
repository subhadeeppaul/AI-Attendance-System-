import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendence' #import our images
images = [] #create a list of images we will import
classNames = [] 
mylist=os.listdir(path)
print(mylist) #name of three images with extensions 
for cls in mylist: #gonna use these names and import the images one by one
    curImg=cv2.imread(f'{path}/{cls}') #read our current image
    images.append(curImg) #append our current image
    classNames.append(os.path.splitext(cls)[0]) 
print(classNames)  #name of the list without extensions

#Start with our encoding process

def findEncodings(images):  
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
    # 3 encoding in our list


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        #print(myDataList)
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')

# initialise the webcam

cap=cv2.VideoCapture(0)


while True:  #To get each frame 1 by 1
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS) #locations
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame) 

    #one by one it will grab one face loc the faces cur frame list
    #it will garb the encoding of encode from encode curr frame 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis =face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


    





