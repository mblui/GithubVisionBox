import os
import cv2
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

##