import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import cv2
import sklearn 
import tensorflow as tf
from tensorflow import keras
from PIL import Image as im
from glob import glob
from sklearn.model_selection import train_test_split
import keras
#from tf.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMAGE_SIZE = [224, 224]

model = keras.models.load_model("VGG16_Garbage_Classifier.h5")

rev_dict = {
    0 : 'Cardboard',
    1 :'Glass', 
    2 : 'Metal',
    3 : 'Paper', 
    4 : 'Plastic',
    5 :'Trash'
}


vid = cv2.VideoCapture(1)
label = False
check = 0

if not (vid.isOpened()):
    print("Could not open video device")

while(True):
    ret, frame = vid.read()
    if cv2.waitKey(33) == ord('a'):
        resized_frame = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(resized_frame)
        img_batch = np.expand_dims(img_array, axis=0)

        img_preprocessed = preprocess_input(img_batch)
        pred = model.predict(img_preprocessed, verbose = 0)
        max_value = np.argmax(pred)
        if max_value >= 0 and max_value <= 4:
            predicition = 'Recycle'
            if(max_value == 0):
                classification = 'Cardboard'
            elif(max_value == 1):
                classification = 'Glass'
            elif(max_value == 2):
                classification = 'Metal'
            elif(max_value == 3):
                classification = 'Paper'
            else:
                classification = 'Plastic'
        else:
            predicition = 'Trash'
        print("It is: ", rev_dict[max_value])
        label = True
        check = 100
    if (label == True):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (0, 20)

        # fontScale
        fontScale = 1

        # Line thickness of 2 px
        thickness = 2

        if predicition == "Recycle":
            # Blue color in BGR
            color = (0, 255, 0)
            # Using cv2.putText() method
            frame = cv2.putText(frame, predicition, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, classification, (0, 470), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
            
        else:
            # Blue color in BGR
            color = (0, 0, 255)
            # Using cv2.putText() method
            frame = cv2.putText(frame, predicition, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('preview',frame)
    if check == 0:
        label = False
    else:
        check = check -1
    if cv2.waitKey(33) == ord('q'):
        break
        
# while(True):
#     ret, frame = vid.read()
#     resized_frame = cv2.resize(frame, (224, 224))
#     img_array = image.img_to_array(resized_frame)
#     img_batch = np.expand_dims(img_array, axis=0)

#     img_preprocessed = preprocess_input(img_batch)
#     pred = model.predict(img_preprocessed, verbose = 0)
#     max_value = np.argmax(pred)
#     if max_value >= 0 and max_value <= 4:
#         predicition = 'Recycle'
#     else:
#         predicition = 'Trash'
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # org
#     org = (0, 20)

#     # fontScale
#     fontScale = 1

#     # Blue color in BGR
#     color = (255, 0, 0)

#     # Line thickness of 2 px
#     thickness = 2

#     # Using cv2.putText() method
#     frame = cv2.putText(frame, predicition, org, font, 
#                    fontScale, color, thickness, cv2.LINE_AA)
#     cv2.imshow('preview',frame)
#     print("It is: ", rev_dict[max_value])
#     if cv2.waitKey(33) == ord('q'):
#         break

vid.release()
cv2.destroyAllWindows()