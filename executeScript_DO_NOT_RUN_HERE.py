## SCRIPT NEEDS TO BE IN FOLDER: /home/jetson/Desktop/librealsense2/wrappers/python/examples/


from subprocess import call
import shutil
import os
import time
import sys
import numpy as np
from git import repo
## Define paths
src_path = r"/home/jetson/Documents/GithHub/GithubVisionBox/"

## Add github path to system path
sys.path.append(src_path)
from config_executeScript import *
file_path = []
ti_m =[]
m_ti =[]

print("##############################")
print("## Start Python Script")

print("## Get latest version of GITHUB FILES--> 0 %")
g = git.cmd.Git(src_path)
g.pull()

# discard any current changes
repo.git.reset('--hard')

# if you need to reset to a specific branch:    
repo.git.reset('--hard','origin/master')

# pull in the changes from from the remote
repo.remotes.origin.pull()
print("## Get latest version of GITHUB FILES --> 100%")

## Load Configuration file
fileRange = file_nr.size 
#print("range is", fileRange)
for fileIndex in range(fileRange):
    #print("fileIndex", fileIndex, 'file nr', file_nr, "available_files", Available_files)
    file_path.append(Available_files[file_nr[fileIndex]])
    ## Last modified state
    ti_m.append(os.path.getmtime(src_path+file_path[fileIndex]))
    m_ti.append(time.ctime(ti_m[fileIndex]))
    print("##", file_path[fileIndex], "was last modified at" , m_ti[fileIndex])
    print("## Loading: ", file_path[fileIndex], "--> 0%")  
    shutil.copy(src_path+file_path[fileIndex], dst_path)
    print("## Loading: ", file_path[fileIndex], "--> 100%")  

#print("Filepath", file_path)
print("## Executing: ", file_path[0])
print("##############################")


os.chdir(dst_path)
print(os.getcwd())

call(["python3", file_path[0]])
