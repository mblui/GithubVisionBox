import cv2
import numpy as np        
import matplotlib.pyplot as plt
from scipy import ndimage
import math
#import tensorflow.keras as keras
#from tensorflow.keras.models import load_model

#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU') 
#for gpu in gpus:
	#tf.config.experimental.set_memory_growth(gpu, True)
	#tf.config.experimental.reset_memory_stats('GPU:0')


# loading pre trained model
#model = load_model('cnn_model/digit_classifier.h5')

#def predict_digit(img):
    #test_image = img.reshape(-1,28,28,1)
    #return np.argmax(model.predict(test_image))


#pitting label
def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 5
    l_y = int(y) + 5
    #cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1) 
    cv2.putText(t_img,str(label),(l_x,l_y), font,0.75,(255,0,0),1,cv2.LINE_AA)
    return t_img

# refining each digit
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray




def get_output_image(path):
  
    img = cv2.imread(path,2)
    img_org =  cv2.imread(path)
    cv2.namedWindow("W1", cv2.WINDOW_NORMAL)
    size = [500,500]
    cv2.imshow('W1',img_org)
    cv2.waitKey(2000)


    ret,thresh = cv2.threshold(img,127,255,0)
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cv2.namedWindow("W2", cv2.WINDOW_NORMAL)
    size = [500,500]
    cv2.resizeWindow("W2", size[0], size[1])
    cv2.imshow('W2',thresh)
    cv2.waitKey(2000)

    # cv2.namedWindow("W1", cv2.WINDOW_NORMAL)
    # size = [500,500]
    # cv2.resizeWindow("W1", size[0], size[1])
    # cv2.imshow('W1',th3)
    # cv2.waitKey(2000)


    for j,cnt in enumerate(contours):
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        
        if(hierarchy[0][j][3]!=-1 and w>20 and h>20):
            #putting boundary on each digit
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),1)
            
            #cropping each image and process
            roi = img[y:y+h, x:x+w]
            roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)
            th,fnl = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
            cv2.namedWindow("W3", cv2.WINDOW_NORMAL)
            size = [500,500]
            cv2.resizeWindow("W3", size[0], size[1])
            cv2.imshow('W3',roi)
            cv2.waitKey(3000)
            # getting prediction of cropped image
            #pred = predict_digit(roi)
            pred = 0.9
            print(pred)
            
            # placing label on each digit
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            img_org = put_label(img_org,pred,x,y)

    return img_org