import pyrealsense2 as rs
import numpy as np
import subprocess
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf

## Defined library locations of images
loc_def_raspberry  = "dgslr@192.168.23.251:/home/dgslr/ProgramFiles/SCP_images/"  
loc_def_jetson = "/home/jetson/Documents/SCP_SharedData/"

## Check GPU & CUDA
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))

print("is_cuda_gpu_available: ", is_cuda_gpu_available)
# Doe hiermee wat je wilt! Succes! 

########################################################
print("## 1) Import Libraries")
print("## Tensorflow version is", tf.__version__)

########################################################
print("## 2) Create customTF2, training and data folders in your google drive")
## Create a folder named customTF2 in your google drive.")
## Create another folder named training inside the customTF2 folder (training folder is where the checkpoints will be saved during training)")
## Create another folder named data inside the customTF2 folder.")

########################################################
print("## 3) Create and upload your image files and xml files.")
# Create a folder named images for your custom dataset images and create another folder named annotations for its corresponding xml files.
# Next, create their zip files and upload them to the customTF2 folder in your drive.
# (Make sure all the image files have extension as ".jpg" only. Other formats like ".png" , ".jpeg" or even ".JPG" will give errors since the generate_tfrecord and xml_to_csv scripts here have only ".jpg" in them)

# Collect Images Dataset and label them to get their PASCAL_VOC XML annotations
## For Datasets, you can check out my Dataset Sources at the bottom of this article in the credits section. You can use any software for labeling like the labelImg tool.
## Read this article to know more about collecting datasets and labeling process.

########################################################
print("## 4) Upload the generate_tfrecord.py file to the customTF2 folder on your drive.")
# Upload the generate_tfrecord.py file to the customTF2 folder on your drive.
# You can find the generate_tfrecord.py file here

########################################################
print(" ## 5) Mount drive and link your folder")

########################################################
print(" ## 6) Clone the tensorflow models git repository & Install TensorFlow Object Detection API")
# clone the tensorflow models on the colab cloud vm
subprocess.run(["git clone --q https://github.com/tensorflow/models.git"])

#navigate to /models/research folder to compile protos
#%cd models/research

# Compile protos.
#protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
#cp object_detection/packages/tf2/setup.py .
#python -m pip install .

print("## END OF FILE")
print("########################################################")


