## SCRIPT NEEDS TO BE IN FOLDER: /home/jetson/Desktop/librealsense2/wrappers/python/examples/


from subprocess import call
import shutil
import os
import time
import sys

## Define paths
src_path = r"/home/jetson/Documents/GithHub/GithubVisionBox/"

## Add github path to system path
sys.path.append(src_path)
from config_executeScript import *

## Load Configuration file
file_path = Available_files[file_nr]

## Last modified state
ti_m = os.path.getmtime(src_path+file_path)
m_ti = time.ctime(ti_m)

print("##############################")
print("## Start Python Script")
print("##", file_path, "was last modified at" , m_ti)
print("## Loading: ", file_path, "--> 0%")  
for file in file_path:
    shutil.copy(src_path+file, dst_path)
print("## Loading: ", file_path, "--> 100%")  
print("## Executing: ", file_path)
print("##############################")


os.chdir(dst_path)
call(["python3", file_path])
