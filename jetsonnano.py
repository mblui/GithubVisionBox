## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# Test EVI

# First import the library's
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess, time 

loc_def_raspberry  = "dgslr@192.168.23.251:/home/dgslr/ProgramFiles/SCP_images/"  
loc_def_jetson = "/home/rddgs/Desktop/Link_to_examples/SCP_SharedData/"


#create variable to store image transfer speeds 
img_tranfer = np.array([])

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Depth Sensor Settings
depth_sensor = device.query_sensors()[0]
depth_sensor.set_option(rs.option.laser_power, 360)			# Infrared Laser Power [0-360 mW]
depth_sensor.set_option(rs.option.depth_units, 0.001)		# Small depth unit = better depth resolution, but less range
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

RGB_sensor = device.query_sensors()[1]
RGB_sensor.set_option(rs.option.brightness, 0)
RGB_sensor.set_option(rs.option.contrast, 50)
RGB_sensor.set_option(rs.option.gamma, 300)
RGB_sensor.set_option(rs.option.hue, 0)
RGB_sensor.set_option(rs.option.saturation, 50)
RGB_sensor.set_option(rs.option.sharpness, 50)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 15 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Count number of images
i = int(0)

# Streaming loop
try:
    while True:
        i=i+1
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = color_image #np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        
        ## Added by Mart #-----------------------------------------------------#
        loc_specific_jetson = loc_def_jetson + "tempImg"+".jpg"
        loc_specific_raspberry = loc_def_raspberry + "img" + str(i) + ".jpg"
        tic = time.perf_counter()
        if not cv2.imwrite(loc_specific_jetson, images):
            break
        tic = time.perf_counter()
        subprocess.run(["scp", loc_specific_jetson, loc_specific_raspberry])     # [{type}, {from directory/file}, {from directory/file}]
        toc = time.perf_counter()
        img_tranfer = np.append(img_tranfer, toc-tic)
        #----------------------------------------------------------------------#

        #cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        #cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
           break
finally:
    pipeline.stop()
    print(f"Downloaded {i:0.1f} files in average {np.average(img_tranfer):0.3f} s ({(1/np.average(img_tranfer)):0.3f}[Hz]) (min/max:{np.min(img_tranfer):0.3f} /{np.max(img_tranfer):0.3f}")