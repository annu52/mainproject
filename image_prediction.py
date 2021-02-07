
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
import cv2

cascade = 'cascade_classifier/face_classifier.xml'
faceCascade =cv2.CascadeClassifier(cascade)
img_size = 224
model = load_model('aimodel/mask_model')
labels_dict = {1:'NO MASK',0:'MASK'}
color_dict  = { 1:(0,0,255),0:(0,255,0)}
frame = cv2.imread('uploads/abc.jpeg')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,1.2,7)

for x,y,w,h in faces:
    face_img = frame[y:y+w,x:x+h]
    resized = cv2.resize(face_img,(img_size,img_size))
    normalized = resized/255.0
    reshaped = np.reshape(normalized,(1,img_size,img_size,3)) 
    result = model.predict(reshaped)
    
    label = np.argmax(result,axis=1)[0]
    print("result",result)
    print('label',label)
    print('face info--------')
    cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
    cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
cv2.imshow('output',frame)
cv2.waitKey(0)