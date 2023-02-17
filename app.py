###############################################################
##########          By: suryaveer @IIT Indore         #########
##########     GITHUB: https://github.com/surya-veer  #########
###############################################################
# import pygame
from  process_image import get_output_image
import os

import cv2
path_2_image = "/home/jetson/Documents/GithHub/GithubVisionBox/customTF2/data/images/"
image_itself= "result_foto14026.jpg_97.jpg"



valid_images = [".jpg",".gif",".png",".tga"]

cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
size = [500,500]
cv2.resizeWindow("Resized_Window", size[0], size[1])

for f in os.listdir(path_2_image):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    path = path_2_image + f
    # print("path =", path)
    img = cv2.imread(path)

    # #cv2.imshow('image',img)
    # #cv2.waitKey(0)
    #cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Resized_Window", size[0], size[1])
    #cv2.imshow('Resized_Window',img)
    #cv2.waitKey(500)
    output = get_output_image(path)
    cv2.imshow('W1',output)
    cv2.waitKey(2000)
    #cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Resized_Window", size[0], size[1])
    #cv2.imshow('Resized_Window',output)
    #cv2.waitKey(2500)


## TODO 
#Add that only black pixels remain black and the rest goes to zero. 