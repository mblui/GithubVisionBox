import os

path_GIT = "/home/jetson/Documents/GithHub/GithubVisionBox/custom_training/"

allfolders = ["test_labels"] #, "train_labels"]

valid_images = [".jpg",".gif",".png",".tga"]

os.chdir(path_GIT)
print("Current directory", os.getcwd())
for f in os.listdir(path_GIT+allfolders[0]):
        print("f", f[:-3])
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