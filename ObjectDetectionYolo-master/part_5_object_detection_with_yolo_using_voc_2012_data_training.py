# -*- coding: utf-8 -*-
"""Part 5 Object Detection with Yolo using VOC 2012 data - training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kvR2Yg98TIhgJaWDlLOO7LIgQnknnLs5

<a data-flickr-embed="true"  href="https://www.flickr.com/photos/157237655@N08/46489714642/in/datetaken-public/" title="YOLO model training in progress"><img src="https://farm8.staticflickr.com/7840/46489714642_d69661a409_b.jpg" width="1024" height="797" alt="YOLO model training in progress"></a><script async src="//embedr.flickr.com/assets/client-code.js" charset="utf-8"></script>

This is the fifth blog post of [Object Detection with YOLO blog series](https://fairyonice.github.io/tag/object-detection-using-yolov2-on-pascal-voc2012-series.html). This blog finally train the model using the scripts that are developed in the [previous blog posts](https://fairyonice.github.io/tag/object-detection-using-yolov2-on-pascal-voc2012-series.html). 
I will use PASCAL VOC2012 data. 
This blog assumes that the readers have read the previous blog posts - [Part 1](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html), [Part 2](https://fairyonice.github.io/Part%202_Object_Detection_with_Yolo_using_VOC_2014_data_input_and_output_encoding.html), [Part 3](https://fairyonice.github.io/Part_3_Object_Detection_with_Yolo_using_VOC_2012_data_model.html), [Part 4](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html).

## Andrew Ng's YOLO lecture
- [Neural Networks - Bounding Box Predictions](https://www.youtube.com/watch?v=gKreZOUi-O0&t=0s&index=7&list=PL_IHmaMAvkVxdDOBRg2CbcJBq9SY7ZUvs)
- [C4W3L06 Intersection Over Union](https://www.youtube.com/watch?v=ANIzQ5G-XPE&t=7s)
- [C4W3L07 Nonmax Suppression](https://www.youtube.com/watch?v=VAo84c1hQX8&t=192s)
- [C4W3L08 Anchor Boxes](https://www.youtube.com/watch?v=RTlwl2bv0Tg&t=28s)
- [C4W3L09 YOLO Algorithm](https://www.youtube.com/watch?v=9s_FpMpdYW8&t=34s)

## Reference
- [You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf) 

- [YOLO9000:Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
 
- [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2)

## Reference in my blog
- [Part 1 Object Detection using YOLOv2 on Pascal VOC2012 - anchor box clustering](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)
- [Part 2 Object Detection using YOLOv2 on Pascal VOC2012 - input and output encoding](https://fairyonice.github.io/Part%202_Object_Detection_with_Yolo_using_VOC_2014_data_input_and_output_encoding.html)
- [Part 3 Object Detection using YOLOv2 on Pascal VOC2012 - model](https://fairyonice.github.io/Part_3_Object_Detection_with_Yolo_using_VOC_2012_data_model.html)
- [Part 4 Object Detection using YOLOv2 on Pascal VOC2012 - loss](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html)
- [Part 5 Object Detection using YOLOv2 on Pascal VOC2012 - training](https://fairyonice.github.io/Part_5_Object_Detection_with_Yolo_using_VOC_2012_data_training.html)
- [Part 6 Object Detection using YOLOv2 on Pascal VOC 2012 data - inference on image](https://fairyonice.github.io/Part_6_Object_Detection_with_Yolo_using_VOC_2012_data_inference_image.html)
- [Part 7 Object Detection using YOLOv2 on Pascal VOC 2012 data - inference on video](https://fairyonice.github.io/Part_7_Object_Detection_with_Yolo_using_VOC_2012_data_inference_video.html)

## My GitHub repository 
This repository contains all the ipython notebooks in this blog series and the funcitons (See backend.py). 
- [FairyOnIce/ObjectDetectionYolo](https://github.com/FairyOnIce/ObjectDetectionYolo)
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import os, sys
# from tensorflow.keras.callbacks import EarlyStopping 
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import SGD 
# from tensorflow.keras.optimizers import Adam 
# from tensorflow.keras.optimizers import RMSprop
# import tensorflow as tf
# print(" ### Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     print("DONE", gpu)
print(sys.version)
# %matplotlib inline

"""## Define anchor box
<code>ANCHORS</code> defines the number of anchor boxes and the shape of each anchor box.
The choice of the anchor box specialization is already discussed in [Part 1 Object Detection using YOLOv2 on Pascal VOC2012 - anchor box clustering](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html). 

Based on the K-means analysis in the previous blog post, I will select 4 anchor boxes of following width and height. The width and heights are rescaled in the grid cell scale (Assuming that the number of grid size is 13 by 13.) See [Part 2 Object Detection using YOLOv2 on Pascal VOC2012 - input and output encoding](https://fairyonice.github.io/Part%202_Object_Detection_with_Yolo_using_VOC_2014_data_input_and_output_encoding.html) to learn how I rescal the anchor box shapes into the grid cell scale.

Here I choose 4 anchor boxes. With 13 by 13 grids, every frame gets 4 x 13 x 13 = 676 bouding box predictions.
"""

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                   10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height

"""## Define Label vector containing 20 object classe names."""

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

"""
## Read images and annotations into memory
Use the pre-processing code for parsing annotation at [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2).
This <code>parse_annoation</code> function is already used in [Part 1 Object Detection using YOLOv2 on Pascal VOC2012 - anchor box clustering](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html) and saved in my python script. 
This script can be downloaded at [my Github repository, FairyOnIce/ObjectDetectionYolo/backend](https://github.com/FairyOnIce/ObjectDetectionYolo/blob/master/backend.py)."""

### The location where the VOC2012 data is saved.
train_image_folder = "/home/jetson/Desktop/VOCdevkit/VOC2012/JPEGImages/"
train_annot_folder = "/home/jetson/Desktop/VOCdevkit/VOC2012/Annotations/"

np.random.seed(1)
from backend import parse_annotation
train_image, seen_train_labels = parse_annotation(train_annot_folder,
                                                  train_image_folder, 
                                                  labels=LABELS)
print("N train = {}".format(len(train_image)))

"""## Instantiate batch generator object
<code>SimpleBatchGenerator</code> is discussed and used in 
[Part 2 Object Detection using YOLOv2 on Pascal VOC2012 - input and output encoding](https://fairyonice.github.io/Part%202_Object_Detection_with_Yolo_using_VOC_2014_data_input_and_output_encoding.html).
This script can be downloaded at [my Github repository, FairyOnIce/ObjectDetectionYolo/backend](https://github.com/FairyOnIce/ObjectDetectionYolo/blob/master/backend.py).
"""

from backend import SimpleBatchGenerator

BATCH_SIZE        = 5
IMAGE_H, IMAGE_W  = 416, 416
GRID_H,  GRID_W   = 13 , 13
TRUE_BOX_BUFFER   = 50
BOX               = int(len(ANCHORS)/2)

generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

print("#"*30)
print("I'm HERE!")

def normalize(image):
    return image / 255.
train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                             norm=normalize, shuffle=True)

"""## Define model
We define a YOLO model.
The model defenition function is already discussed in [Part 3 Object Detection using YOLOv2 on Pascal VOC2012 - model](https://fairyonice.github.io/Part_3_Object_Detection_with_Yolo_using_VOC_2012_data_model.html) and all the codes are available at [my Github](https://github.com/FairyOnIce/ObjectDetectionYolo/blob/master/backend.py).
"""

from backend import define_YOLOv2, set_pretrained_weight, initialize_weight
CLASS             = len(LABELS)
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                  trainable=False)
model.summary()
print("#"*30)
print("model.summary!")
"""## Initialize the weights
The initialization of weights are already discussed in [Part 3 Object Detection using YOLOv2 on Pascal VOC2012 - model](https://fairyonice.github.io/Part_3_Object_Detection_with_Yolo_using_VOC_2012_data_model.html). 
All the codes from [Part 3](https://fairyonice.github.io/Part_3_Object_Detection_with_Yolo_using_VOC_2012_data_model.html) are stored at [my Github](https://github.com/FairyOnIce/ObjectDetectionYolo/blob/master/backend.py).
"""

# #path_to_weight = "./yolov2.weights"
# path_to_weight = "/home/jetson/Documents/GithHub/GithubVisionBox/ObjectDetectionYolo-master/yolov2.weights"
# nb_conv        = 22
# model          = set_pretrained_weight(model,nb_conv, path_to_weight)
# layer          = model.layers[-4] # the last convolutional layer
# initialize_weight(layer,sd=1/(GRID_H*GRID_W))

# """## Loss function
# We already discussed the loss function of YOLOv2 implemented by [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2) in [Part 4 Object Detection using YOLOv2 on Pascal VOC2012 - loss](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html).
# I modified the codes and the codes are available at [my Github](https://github.com/FairyOnIce/ObjectDetectionYolo/blob/master/backend.py).
# """

# from backend import custom_loss_core 
# #help(custom_loss_core)
# print("#"*30)
# print("imported custom_loss_core!")

# """Notice that this custom function <code>custom_loss_core</code> depends not only on <code>y_true</code> and <code>y_pred</code> but also the various hayperparameters.
# Unfortunately, Keras's loss function API does not accept any parameters except <code>y_true</code> and <code>y_pred</code>. Therefore, these hyperparameters need to be defined globaly. 
# To do this, I will define a wrapper function <code>custom_loss</code>.
# """
# print("#"*30)
# print("I'm HERE2!")
# GRID_W             = 13
# GRID_H             = 13
# BATCH_SIZE         = 34
# LAMBDA_NO_OBJECT = 1.0
# LAMBDA_OBJECT    = 5.0
# LAMBDA_COORD     = 1.0
# LAMBDA_CLASS     = 1.0
    
# def custom_loss(y_true, y_pred):
#     return(custom_loss_core(
#                      y_true,
#                      y_pred,
#                      true_boxes,
#                      GRID_W,
#                      GRID_H,
#                      BATCH_SIZE,
#                      ANCHORS,
#                      LAMBDA_COORD,
#                      LAMBDA_CLASS,
#                      LAMBDA_NO_OBJECT, 
#                      LAMBDA_OBJECT))

# """## Training starts here! 
# Finally, we start the training here.
# We only train the final 23rd layer and freeze the other weights.
# This is because I am unfortunately using CPU environment.
# """


# print("#"*30)
# print("I'm HERE1!")

# dir_log = "logs/"
# try:
#     os.makedirs(dir_log)
# except:
#     pass


# BATCH_SIZE   = 4
# generator_config['BATCH_SIZE'] = BATCH_SIZE
# early_stop = EarlyStopping(monitor='loss', 
#                            min_delta=0.001, 
#                            patience=3, 
#                            mode='min', 
#                            verbose=1)
# checkpoint = ModelCheckpoint('weights_yolo_on_voc2012.h5', 
#                              monitor='loss', 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode='min',
#                              save_freq='epoch')

# optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# #optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
# #optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
# #tf.compat.v1.disable_eager_execution()
# print("#"*30)
# print("I'm HERE!")
# model.compile(loss=custom_loss, optimizer=optimizer, experimental_run_tf_function=False, run_eagerly=True)
# model.run_eagerly = True
# model.fit(train_batch_generator, 
#             steps_per_epoch  = len(train_batch_generator), 
#             epochs           = 1, 
#             verbose          = 1,
#             #validation_data  = valid_batch,
#             #validation_steps = len(valid_batch),
#             callbacks        = [early_stop, checkpoint], 
#             max_queue_size   = 1)

# # """[FairyOnIce/ObjectDetectionYolo](https://github.com/FairyOnIce/ObjectDetectionYolo)
# #  contains this ipython notebook and all the functions that I defined in this notebook. 

# # By accident, I stopped a notebook.
# # Here, let's resume the training..
# # """

# # model.load_weights('weights_yolo_on_voc2012.h5')
# # model.fit_generator(generator        = train_batch_generator, 
# #                     steps_per_epoch  = len(train_batch_generator), 
# #                     epochs           = 50, 
# #                     verbose          = 1,
# #                     #validation_data  = valid_batch,
# #                     #validation_steps = len(valid_batch),
# #                     callbacks        = [early_stop, checkpoint], 
# #                     max_queue_size   = 3)

