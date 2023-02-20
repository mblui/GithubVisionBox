import os
import shutil

## Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ## Import once labels and files
# if 0:
#     path_GIT = "/home/jetson/Documents/GithHub/GithubVisionBox/custom_training/"
#     allfolders = ["JPEGImages/", "data_in/test_labels", "data_in/", "data_in/train_labels", "data_in/"]
#     # hoi
#     valid_images = [".jpg",".gif",".png",".tga"]
#     i = 0
#     os.chdir(path_GIT)
#     print("Current directory", os.getcwd())
#     for f in os.listdir(path_GIT+allfolders[3]):
#         #print("XML FILE", f)
#         for ff in os.listdir(path_GIT+allfolders[0]):
#             #print("image", ff)
#             if f[:-4] == ff[:-4]:
#                 destimation = path_GIT + allfolders[2] + ff
#                 origin = path_GIT +  allfolders[0] + ff
#                 print("destimation", destimation)
#                 print("origin", origin)
#                 cv2.imshow('hoi', cv2.imread(origin,2))
#                 cv2.waitKey(150)
#                 #print("destimation", destimation)
#                 #print("origin", origin)
#                 shutil.copyfile(origin, destimation)
#                 i = i + 1
#     print("Total matches =", i)

###############################################################

import numpy as np
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU') 
#for gpu in gpus:
#	tf.config.experimental.set_memory_growth(gpu, True)
        
#import matplotlib.pyplot as plt
#import tensorflow.keras as keras
#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications import VGG16
#import os 
import xml.etree.ElementTree as ET
from collections import OrderedDict
#import matplotlib.pyplot as plt
import pandas as pd 


customFolder = "custom_training/result/"
dir_anno = customFolder + "Annotations"
img_dir  = customFolder + "JPEGImages"

def extract_single_xml_file(tree):
    Nobj = 0
    row  = OrderedDict()
    for elems in tree.iter():

        if elems.tag == "size":
            for elem in elems:
                row[elem.tag] = int(elem.text)
        if elems.tag == "object":
            for elem in elems:
                if elem.tag == "name":
                    row["bbx_{}_{}".format(Nobj,elem.tag)] = str(elem.text)              
                if elem.tag == "bndbox":
                    for k in elem:
                        row["bbx_{}_{}".format(Nobj,k.tag)] = float(k.text)
                    Nobj += 1
    row["Nobj"] = Nobj
    return(row)

df_anno = []
for fnm in os.listdir(dir_anno):  
    if not fnm.startswith('.'): ## do not include hidden folders/files
        tree = ET.parse(os.path.join(dir_anno,fnm))
        row = extract_single_xml_file(tree)
        row["fileID"] = fnm.split(".")[0]
        df_anno.append(row)
df_anno = pd.DataFrame(df_anno)

maxNobj = np.max(df_anno["Nobj"])


print("columns in df_anno\n-----------------")
for icol, colnm in enumerate(df_anno.columns):
    print("{:3.0f}: {}".format(icol,colnm))
print("-"*30)
print("df_anno.shape={}=(N frames, N columns)".format(df_anno.shape))
df_anno.head()