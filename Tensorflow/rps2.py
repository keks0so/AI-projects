import cv2 
import os
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import scipy

model_rps =  tf.keras.models.load_model('rps.h5')

TRAINING_DIR = "C:/Users/Admin/Desktop/JUPITER/archive"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "C:/Users/Admin/Desktop/JUPITER/rps-cv-images"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

vid = cv2.VideoCapture(0)



def rps():
    model_rps =  tf.keras.models.load_model('rps.h5')
    
    yhat = model_rps.predict(np.expand_dims(resize/255, 0))
    huy = yhat[0]
    rez = huy.min()
    rez1 = str(rez)
    rez2 = rez1[0]+ rez1[1] + rez1[2]+ rez1[3]
    rez = float(rez2)
    if (rez >= 1) and (rez < 2): 
        print(f'Predicted class is rock')
        
    if (rez >= 2) and (rez < 3):
        print(f'Predicted class is paper')
        
    if (rez >= 3) and (rez < 4):
        print(f'Predicted class is scisors')
        

while True:
    

    model_rps =  tf.keras.models.load_model('rps.h5')
    
    
    
    

    success, img = vid.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

    resize = tf.image.resize(img_rgb, (150,150))
    resize = tf.image.resize(img, (150,150))
    yhat = model_rps.predict(np.expand_dims(resize/255, 0))
    rps()

    

    cv2.imshow("Rock Paper Scisors", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break