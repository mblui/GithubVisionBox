import os
import shutil


## Import once labels and files
if 0:
    path_GIT = "/home/jetson/Documents/GithHub/GithubVisionBox/custom_training/"

    allfolders = ["JPEGImages/", "data_in/test_labels", "data_in/", "data_in/train_labels", "data_in/"]
    # hoi
    valid_images = [".jpg",".gif",".png",".tga"]
    i = 0
    os.chdir(path_GIT)
    print("Current directory", os.getcwd())
    for f in os.listdir(path_GIT+allfolders[3]):
        #print("XML FILE", f)
        for ff in os.listdir(path_GIT+allfolders[0]):
            #print("image", ff)
            if f[:-4] == ff[:-4]:
                destimation = path_GIT + allfolders[2] + ff
                origin = path_GIT +  allfolders[0] + ff
                print("destimation", destimation)
                print("origin", origin)
                cv2.imshow('hoi', cv2.imread(origin,2))
                cv2.waitKey(150)
                #print("destimation", destimation)
                #print("origin", origin)
                shutil.copyfile(origin, destimation)
                i = i + 1
    print("Total matches =", i)

###############################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPool2D
# from keras import optimizers
# from keras.datasets import mnist
# from keras.utils import to_categorical

import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications import VGG16


# import cv2
# import pandas as pd
# import os, sys 
# #import scipy.misc
# import matplotlib.pyplot as plt
# #import random
# import imageio
# #import skimage

modelvgg16 = VGG16(include_top=True,weights='imagenet')
modelvgg16.summary()

