import os
import cv2
import shutil
path_GIT = "/home/jetson/Documents/GithHub/GithubVisionBox/custom_training/"

allfolders = ["JPEGImages/", "data_in/test_labels", "data_in/", "data_in/train_labels", "data_in/"]
# hoi
valid_images = [".jpg",".gif",".png",".tga"]
i = 0
os.chdir(path_GIT)
print("Current directory", os.getcwd())
for f in os.listdir(path_GIT+allfolders[1]):
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

    #ext = os.path.splitext(f)[1]
    #if ext.lower() not in valid_images:
    #    continue
    #path = path_2_image + f
    ## print("path =", path)
    #img = cv2.imread(path)

    # #cv2.imshow('image',img)
    # #cv2.waitKey(0)
    #cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Resized_Window", size[0], size[1])
    #cv2.imshow('Resized_Window',img)
    #cv2.waitKey(500)
    #output = get_output_image(path)
    #cv2.imshow('W3',output)
    #cv2.waitKey(3000)
    #cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Resized_Window", size[0], size[1])
    #cv2.imshow('Resized_Window',output)
    #cv2.waitKey(2500)


## TODO 
#Add that only black pixels remain black and the rest goes to zero. 