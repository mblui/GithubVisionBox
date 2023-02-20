import os
import shutil

## Disable GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
        
import matplotlib.pyplot as plt
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

path_2_csv = customFolder + "df_anno.csv"
def extract_single_xml_file(tree):
    Nobj = 0
    row  = OrderedDict()
    for elems in tree.iter():
        if elems.tag == "size":
            for elem in elems:
                ######################################################################### WORKAROUND HERE! 
                val = elem.text if elem.text else 1     ## depth in the VOC is empty. int() will complain about it!
                row[elem.tag] = int(val)
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

if False: #os.path.exists(path_2_csv):
    print("File is already been created earlier")
else:
    df_anno = []
    for fnm in os.listdir(dir_anno):  
        if not fnm.startswith('.'): ## do not include hidden folders/files
            #print("FNM", fnm)
            tree = ET.parse(os.path.join(dir_anno,fnm))
            #print("Tree", tree)
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

    dir_preprocessed = customFolder
    df_anno.to_csv(os.path.join(dir_preprocessed,"df_anno.csv"),index=False)
#############################################
#plt.hist(df_anno["Nobj"].values,bins=100)
#plt.title("max N of objects per image={}".format(maxNobj))
#plt.show()


####################################################
## Show how classes are distributed
    
from collections import Counter
class_obj = []
for ibbx in range(maxNobj):
    class_obj.extend(df_anno["bbx_{}_name".format(ibbx)].values)
class_obj = np.array(class_obj)

count             = Counter(class_obj[class_obj != 'nan'])
print(count)
class_nm          = list(count.keys())
class_count       = list(count.values())
asort_class_count = np.argsort(class_count)

class_nm          = np.array(class_nm)[asort_class_count]
class_count       = np.array(class_count)[asort_class_count]

xs = range(len(class_count))
plt.barh(xs,class_count)
plt.yticks(xs,class_nm)
plt.title("The number of objects per class: {} objects in total".format(len(count)))
plt.show()
####################################################
## SHOW arbitrary image with label
import imageio
def plt_rectangle(plt,label,x1,y1,x2,y2):
    '''
    == Input ==
    
    plt   : matplotlib.pyplot object
    label : string containing the object class name
    x1    : top left corner x coordinate
    y1    : top left corner y coordinate
    x2    : bottom right corner x coordinate
    y2    : bottom right corner y coordinate
    '''
    linewidth = 3
    color = "yellow"
    plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)
    
# randomly select 20 frames    
size = 20    
ind_random = np.random.randint(0,df_anno.shape[0],size=size)
for irow in ind_random:
    print("irow", irow)
    row  = df_anno.iloc[irow,:]
    path = os.path.join(img_dir, row["fileID"] + ".jpg")
    # read in image
    img  = imageio.imread(path)

    plt.figure(figsize=(12,12))
    plt.imshow(img) # plot image
    plt.title("Nobj={}, height={}, width={}".format(row["Nobj"],row["height"],row["width"]))
    # for each object in the image, plot the bounding box
    for iplot in range(row["Nobj"]):
        plt_rectangle(plt,
                      label = row["bbx_{}_name".format(iplot)],
                      x1=row["bbx_{}_xmin".format(iplot)],
                      y1=row["bbx_{}_ymin".format(iplot)],
                      x2=row["bbx_{}_xmax".format(iplot)],
                      y2=row["bbx_{}_ymax".format(iplot)])
    plt.show() ## show the plot
