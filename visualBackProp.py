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
from keras.layers import merge


import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray
from PIL import Image, ImageEnhance, ImageOps

import matplotlib.cm as cm
from keras.models import Model


def almostEquals(a,b,thres=50):
    return all(abs(a[i]-b[i])<thres for i in range(len(a)))


#
#  Create the models
#

def convertToNewModel(name):
  model = load_model(name)
  model.summary()

  # grab the conv layers
  current_stack=[]
  for layer in model.layers:
    if layer.name.startswith("conv"):
        current_stack.insert(0, layer)
  # grab the activation layers connected to the conv layers
  # TODO - this is a bad way to do it!!
  act_stack=[]
  for layer in model.layers:
    if layer.name.startswith("activ"):
        act_stack.insert(0, layer)
  start=len(act_stack)-len(current_stack)
  act_stack=act_stack[start:]

  lastone=None
  #  hold onto last one..
  for i, layer in enumerate(current_stack):
    print(layer.name,i)
    our_shape=(layer.output_shape[1],layer.output_shape[2],1)
    hidden_layer = act_stack[i]
    print(hidden_layer.name)
    print(layer.name)
    print(our_shape)
    # average this layer
    name='lambda_new_'+str(i)
    c1=Lambda(lambda xin: K.mean(xin,axis=3),name=name)(hidden_layer.output)
    name='reshape_new_'+str(i)
    r1=Reshape(our_shape,name=name)(c1)
    lastone=r1
    if (i!=0):
       # if we aren't the bottom, multiply by output of layer below
       print("multiply")
       name='multiply_'+str(i)
       r1 = merge([r1,lastone], mode='mul', name = name)
       lastone=r1
    
    
    if (i<len(current_stack)-1):
        print('do deconv')
        # deconv to the next bigger size
        bigger_shape=current_stack[i+1].output_shape
    else:
        bigger_shape=model.input_shape
            
            
    bigger_shape=(bigger_shape[0],bigger_shape[1],bigger_shape[2],1)

    subsample=current_stack[i].subsample
    print(subsample)
    nb_row=current_stack[i].nb_row
    nb_col=current_stack[i].nb_col
    print(nb_col,nb_row)
    print(bigger_shape)
    name='deconv_new_'+str(i)
    d1 = Deconvolution2D(1, nb_row, nb_col,output_shape=bigger_shape,subsample= subsample,border_mode='valid',activation='relu',init='one',name=name)(r1)
    #d4 = Deconvolution2D(1, 3, 3,output_shape=(None,4,17,1),subsample= (2, 2),border_mode='valid',activation='relu',init='one')(r4)

    lastone=d1


  model2 = Model(input=model.input,output=[lastone])
  model2.summary()

  return model2


def processFrame(image,model2):

    #steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
    #   print(image.shape)
    m1d = model2.predict(image[None, :, :, :], batch_size=1)
    #print(m1d.shape)
    m1d = np.squeeze(m1d, axis=0)
    m1d = np.squeeze(m1d, axis=2)
    #print(m1d.shape)

    #print(m1d)
    #plt.hist(m1d[::-1])
    #plt.show()
    #print(m1d.max())
    #print(m1d.min())
    o2=overlay = Image.fromarray(cm.Reds(m1d/m1d.max(), bytes=True)) 
    #plt.imshow(o2);
    #plt.show();

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
    #plt.imshow(o2);
    #plt.show();
    return new_img2


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
    global model
    print(files[count])
    car=np.array(Image.open(files[count]))
    car=car[60:,:]
    count=count+1
    out=processFrame(car,model)
    return  np.array(out)
#    return  car

files=loadTraining()
model=convertToNewModel("model-20.h5")

clip = mpy.VideoClip(make_frame, duration=90) # 2 seconds
clip.write_videofile("out.mp4",audio=False,fps=30)
#clip.write_gif("out.gif",fps=24)


