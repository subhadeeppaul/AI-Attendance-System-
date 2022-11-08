import cv2
import numpy as np
import face_recognition

#Step-1 loading the images and coverting it into RGB

imgElon= face_recognition.load_image_file('ImagesBasics/Elon Musk.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB) #convert it into RGB
#Test Image
imgTest= face_recognition.load_image_file('ImagesBasics/Elon Test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#step-2 finding the faces in our images 

faceLoc=face_recognition.face_locations(imgElon)[0] #detect face
encodeElon=face_recognition.face_encodings(imgElon)[0] 
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Step-3 comparing those faces and finding the distance between them (final Step)
#backend Linear SVM wheather they matched or not

results=face_recognition.compare_faces([encodeElon],encodeTest)
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
#Result-> True or False and distance lower better match


#put_Text=True or false in image
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgElon )
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)