import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess, time 
import tensorflow as tf

## Defined library locations of images
loc_def_raspberry  = "dgslr@192.168.23.251:/home/dgslr/ProgramFiles/SCP_images/"  
loc_def_jetson = "/home/jetson/Documents/SCP_SharedData/"

## Check GPU & CUDA
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))

print("is_cuda_gpu_available: ", is_cuda_gpu_available)
print("is_cuda_gpu_min_3: ", is_cuda_gpu_min_3)

# Doe hiermee wat je wilt!

