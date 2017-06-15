from keras.models import *
from keras.callbacks import *
from keras.layers import Lambda, Convolution2D, Activation, Dropout, Flatten, Dense
from keras.layers import Dense, Lambda, ELU
from keras.layers import Dense, Activation, Reshape, Merge
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras.layers import Merge
import keras.backend as K
import cv2
import argparse
import pickle
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Activation, Lambda, Input, Deconvolution2D, Flatten, Dense, Reshape
from keras.layers import Merge
from keras.models import Sequential
from keras import backend as K


import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray
from PIL import Image, ImageEnhance, ImageOps

def almostEquals(a,b,thres=50):
    return all(abs(a[i]-b[i])<thres for i in range(len(a)))

import matplotlib.cm as cm


#
#  Create the models
#
model = load_model("model-20.h5")
model.summary()



from keras.models import Model
hidden_layer = model.layers[2].output
c1=Lambda(lambda xin: K.mean(xin,axis=3),name='lambda_new_1')(hidden_layer)
r1=Reshape((28,78,1))(c1)
d1 = Deconvolution2D(1, 5, 5,output_shape=(None,60,160, 1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(r1)

model2 = Model(input=model.input,output=[c1,d1])
model2.summary()

a = Input(shape=(28,78,1))
d1 = Deconvolution2D(1, 5, 5,output_shape=(None,60,160, 1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(a)
model2b = Model(input=a,output=[d1])
model2b.summary()


hidden_layer = model.layers[4].output
c2=Lambda(lambda xin: K.mean(xin,axis=3),name='lambda_new_2')(hidden_layer)
r2=Reshape((12,37,1))(c2)
d2 = Deconvolution2D(1, 5, 5,output_shape=(None,28,78,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(r2)

model3 = Model(input=model.input,output=[c2,d2])
model3.summary()

a = Input(shape=(12,37,1))
d2 = Deconvolution2D(1, 5, 5,output_shape=(None,28,78,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(a)
model3b = Model(input=a,output=[d2])
model3b.summary()


hidden_layer = model.layers[6].output
c3=Lambda(lambda xin: K.mean(xin,axis=3),name='lambda_new_3')(hidden_layer)
r3=Reshape((4,17,1))(c3)
d3 = Deconvolution2D(1, 5, 5,output_shape=(None,12,37,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(r3)

model4 = Model(input=model.input,output=[c3,d3])
model4.summary()

a = Input(shape=(4,17,1))
d3 = Deconvolution2D(1, 5, 5,output_shape=(None,12,37,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(a)
model4b = Model(input=a,output=[d3])
model4b.summary()




hidden_layer = model.layers[8].output
c4=Lambda(lambda xin: K.mean(xin,axis=3),name='lambda_new_4')(hidden_layer)
r4=Reshape((1,8,1))(c4)
d4 = Deconvolution2D(1, 3, 3,output_shape=(None,4,17,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(r4)

model5 = Model(input=model.input,output=[c4,d4])
model5.summary()


def processFrame(image):
   global model
   global model2
   global model3
   global model4
   global model5
   global model2b
   global model3b
   global model4b
   print("processFrame")
   print(image.shape)
   #image_batch= np.expand_dims(image,axis=0)
   steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
   print("after predict")
   print(steering_angle)
   conv_cat2 = model2.predict(image[None, :, :, :], batch_size=1)
   conv_cat3 = model3.predict(image[None, :, :, :], batch_size=1)
   conv_cat4 = model4.predict(image[None, :, :, :], batch_size=1)
   conv_cat5 = model5.predict(image[None, :, :, :], batch_size=1)

   deconv5 = conv_cat5[1]
   deconv5 = np.squeeze(deconv5, axis=0)
   deconv5 = np.squeeze(deconv5, axis=2)

   conv_cat2 = np.squeeze(conv_cat2[0], axis=0)
   conv_cat3 = np.squeeze(conv_cat3[0], axis=0)
   conv_cat4 = np.squeeze(conv_cat4[0], axis=0)
   conv_cat5 = np.squeeze(conv_cat5[0], axis=0)

   m3=np.multiply(deconv5,conv_cat4)
   m3d = model4b.predict(m3[None, :, :, None], batch_size=1)
   m3d = np.squeeze(m3d, axis=0)
   m3d = np.squeeze(m3d, axis=2)
   m2=np.multiply(m3d,conv_cat3)
   m2d = model3b.predict(m2[None, :, :, None], batch_size=1)
   m2d = np.squeeze(m2d, axis=0)
   m2d = np.squeeze(m2d, axis=2)
   m1=np.multiply(m2d,conv_cat2)
   m1d = model2b.predict(m1[None, :, :, None], batch_size=1)
   print(m1d.shape)
   m1d = np.squeeze(m1d, axis=0)
   m1d = np.squeeze(m1d, axis=2)
   print(m1d.shape)
   print(m1d.max())
   print(m1d.min())

   #o2=overlay = Image.fromarray(cm.Reds(m1d/255, bytes=True)) 
   o2=overlay = Image.fromarray(cm.Reds(m1d/m1d.max(), bytes=True)) 

   pixeldata = list(overlay.getdata())

   for i,pixel in enumerate(pixeldata):
    if almostEquals(pixel[:3], (255,255,255)):
        pixeldata[i] = (255,255,255,0)
    else:
        pixeldata[i]= (pixel[0],pixel[1],pixel[2],128)

   overlay.putdata(pixeldata)
   carimg = Image.fromarray(np.uint8(image))
   carimg = carimg.convert("RGBA")
   new_img2=Image.alpha_composite(carimg, overlay)
   new_img2= new_img2.convert("RGB")
   o2= o2.convert("RGB")

   return new_img2
   #return carimg
   #return o2

import os
def getFiles(name):
    files = os.listdir(name)
    files = [f for f in files if f[-3:] =='jpg']
    files.sort()
    file_paths = [os.path.join(name, f) for f in files]
    return file_paths

def loadTraining():
    inputs='/home/alans/shark/log/'
    files=getFiles(inputs)
    return files

import moviepy.editor as mpy

count = 0
def make_frame(t):
    global count
    print(files[count])
    car=np.array(Image.open(files[count]))
    car=car[60:,:]
    count=count+1
    out=processFrame(car)
    return  np.array(out)
#    return  car

files=loadTraining()

clip = mpy.VideoClip(make_frame, duration=90) # 2 seconds
clip.write_videofile("out.mp4",audio=False,fps=30)
#clip.write_gif("out.gif",fps=24)


