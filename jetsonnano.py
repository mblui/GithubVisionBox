# load all packages
import pyrealsense2 as rs				# intel Realsense2 camera wrapper
import matplotlib.pyplot as plt
import numpy as np
import cv2								# OpenCV
import sys
import os
import time
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from time import process_time, time
from timeit import default_timer as timer
from termcolor import colored

#hoi TEST 5 if autoupdate works
# Load separate python programs
from UserInputs import *
from find_orientation_and_CoM import *
from FindOrientationTopLayer import *
from ObtainImages import *
from convert_to_3D_point import *
from object_classification import *
from Estimate_depth_distribution import *
from convert_cam_to_otherFrames import *
from convert_to_offset_spacerCoM import *

sys.path.append(os.path.abspath("/home/dgs-lr1/librealsense2/wrappers/python/examples"))

# Create pipeline
Start_timer = timer()
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Depth Sensor Settings
depth_sensor = device.query_sensors()[0]
depth_sensor.set_option(rs.option.laser_power, 360)			# Infrared Laser Power [0-360 mW]
depth_sensor.set_option(rs.option.depth_units, 0.0001)		# Small depth unit = better depth resolution, but less range
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

RGB_sensor = device.query_sensors()[1]
RGB_sensor.set_option(rs.option.brightness, 0)
RGB_sensor.set_option(rs.option.contrast, 50)
RGB_sensor.set_option(rs.option.gamma, 300)
RGB_sensor.set_option(rs.option.hue, 0)
RGB_sensor.set_option(rs.option.saturation, 50)
RGB_sensor.set_option(rs.option.sharpness, 50)

# Set resolution of RGB and DEPTH sensor
config.enable_stream(rs.stream.depth, DEPTH_res[0], DEPTH_res[1], rs.format.z16, 30)
config.enable_stream(rs.stream.color, RGB_res[0], RGB_res[1], rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Prepare align operator
align_to = rs.stream.color
align = rs.align(align_to)

# Count Variable
FinalCountdown = 1

# Variables for accuracy measurement
LoadInitials_timer = timer()
numberOfFind_Orientation_Errors = 0
FatalError = 0

# Load once the figure for plotting result
plt.rcParams["figure.figsize"] = (5,6)
plt.figure()

while_loop_timer_append = []
timing_stop_objectclass_append = []
timing_stop_findOrient_append = []
timing_convertFrames_append = []
timing_load_images_append = []
timing_stop_depthEstimation_append = []

keepArea = []
try:
	while True:
		''' Step 1: Capture image and post-process '''
		# Timing Analysis
		if DoTimingAnalysis == 1:
			timing_start_while = timer()
			mart = 0

		# Obtain Images
		depth_image, color_image, depth_frame = ObtainImages(pipeline, align, profile, depth_scale, MakePlots=0)

		# Timing Analysis
		if DoTimingAnalysis == 1:
			timing_stop_imageLoad = timer()
			timing_load_images_append.append(timing_stop_imageLoad-timing_start_while)
			timing_start_objectclass = timer()

		''' Step 2: Determine whether object is spacer, box layer or obstacle '''
		ObjectClass, contour, DepthOfClosestLayer, DepthOfClosestLayer_corrected, area, depth_image = object_classification(depth_image, depth_frame, color_image, depth_scale, Makeplots=0)

		# Timing Analysis
		if DoTimingAnalysis == 1:
			timing_stop_objectclass = timer()
			timing_stop_objectclass_append.append(timing_stop_objectclass-timing_start_objectclass)
			timing_start_DepthEstimation = timer()

		''' Step 3: Top layer identification '''
		theta, phi, depth_plane, coefficients = Estimate_depth_distribution(depth_image, depth_frame, DepthOfClosestLayer_corrected, depth_scale, Makeplots=0)

		# Timing Analysis
		if DoTimingAnalysis == 1:
			timing_stop_DepthEstimation = timer()
			timing_stop_depthEstimation_append.append(timing_stop_DepthEstimation - timing_start_DepthEstimation)

		''' Step 4: Validate theta, phi and filter out obstacles '''
		# Compare to allowed psi and phi set + obstacles
		if np.abs(theta) > MaximumAllowedTiltAngle or np.abs(phi) > MaximumAllowedTiltAngle or ObjectClass == "obstacle":
			if ObjectClass == "obstacle":
				print("ObjectClass:", colored(ObjectClass, 'red'), ", Area:", area, "[m2]. ") #, colored("Rest of iteration is skipped!", 'red'))
			else:
				print(colored("ERROR: Spacer out of reach. Rest of script has been skipped!", 'red'))
			plt.waitforbuttonpress(0)
			plt.close('all')
			continue

		''' Step 5: Detect orientation top layer  '''
		# If spacer is detected, detect orientation top layer
		if ObjectClass == "spacer":
			# Timing Analysis
			if DoTimingAnalysis == 1:
				timing_start_findOrient = timer()
				mart = 0

			# Detect horizontal orientation (theta) and Centre of Mass
			centrePoint, psi = find_orientation_and_CoM(depth_image, color_image, depth_frame, DepthOfClosestLayer, depth_scale, FinalCountdown, theta, phi, MakePlots=0)

			if np.abs(psi) > MaximumAllowedPsiAngle - 1:
				print(colored(("ERROR: Psi angle is outside valid range of [{:.1f}, {:.1f}] in [degrees]".format(-MaximumAllowedPsiAngle, MaximumAllowedPsiAngle)), 'red'))
				continue

			# If no contour is found, skip iteration and try again.
			# If this occurs 5 times in a row, then let the script stop and inform operator.
			if centrePoint[0] == 0 and centrePoint[1] == 0 and psi == 0:
				print(colored("ERROR: Problem in find_orientation_and_CoM. Capture new images.", 'red'))
				numberOfFind_Orientation_Errors = numberOfFind_Orientation_Errors + 1
				if numberOfFind_Orientation_Errors > 500:
					print(colored("FATAL ERROR: Too often problems in find_orientation_and_CoM. Ensure situation is 'normal'", 'red'))
					FatalError = 1
					break
				continue
			numberOfFind_Orientation_Errors = 0

			# Timing analysis
			if DoTimingAnalysis == 1:
				timing_stop_findOrient = timer()
				timing_stop_findOrient_append.append(timing_stop_findOrient - timing_start_findOrient)
				timing_start_convertFrames = timer()

			''' Step 6: Validate set points and convert to 3D point in mm '''
			# Convert from pixel to CoM of spacer as 3D point in camera frame
			CamFrame = convert_to_3D_point(depth_frame, depth_image, depth_scale, centrePoint, DepthOfClosestLayer)

			''''Step 7: Calculate 3D setpoint x-mm above plane '''
			coefficients = np.array([coefficients[0], coefficients[1], 1, coefficients[2]])

			newPoints_x = +t * coefficients[0] + CamFrame[0] / 1000
			newPoints_y = +t * coefficients[1] + CamFrame[1] / 1000
			newPoints_z = -t * coefficients[2] + CamFrame[2] / 1000
			newPoint = np.array([newPoints_x, newPoints_y, newPoints_z])
			CamFrame = newPoint * 1000

			# Correct depth by tilting
			point = [offsetGripper, 0, 0]
			theta_rad = np.deg2rad(theta)
			phi_rad = np.deg2rad(phi)
			psi_rad = np.deg2rad(psi)

			rx = np.array([[1, 0, 0],
						   [0, np.cos(theta_rad), -np.sin(theta_rad)],
						   [0, np.sin(theta_rad), np.cos(theta_rad)]])
			ry = np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)],
						   [0, 1, 0],
						   [-np.sin(phi_rad), 0, np.cos(phi_rad)]])
			rz = np.array([[np.cos(psi_rad), -np.sin(psi_rad), 0],
						   [np.sin(psi_rad), np.cos(psi_rad), 0],
						   [0, 0, 1]])
			rout = np.dot(rz, np.dot(ry, rx))
			output = np.matmul(rout, point)
			CamFrame[2] = CamFrame[2] + output[2]

			# Convert from CoM of spacer to CoM of gripper as 3D point in camera frame
			CamFrame_offCentre = convert_to_offset_spacerCoM(CamFrame, psi)

			# Convert CoM of gripper as 3D point in camera frame to base- and robot frame
			BaseFrame, robotFrame = convert_cam_to_otherFrames(CamFrame_offCentre, DepthOfClosestLayer_corrected)

			# Timing analysis
			if DoTimingAnalysis == 1:
				timing_stop_convertFrames = timer()
				timing_convertFrames_append.append(timing_stop_convertFrames - timing_start_convertFrames)

		# Timing analysis
		if DoTimingAnalysis == 1:
			timing_end_while = timer()
			while_loop_timer_append.append(timing_end_while-timing_start_while)

		# Print all output variables
		print("Object:", ObjectClass, ", Area  :", area, "[m^2]", ", CycleTime:", np.round(timing_end_while - timing_start_while, 2), "[s]")
		print("Pos X :", np.round(BaseFrame[0], 2), "[mm]")
		print("Pos Y :", np.round(BaseFrame[1], 2), "[mm]")
		print("Pos Z :", np.round(BaseFrame[2]+DepthOfClosestLayer_corrected, 2), "[mm]")
		print("theta :", np.round(theta, 3), "[degrees]")
		print("phi   :", np.round(phi, 3), "[degrees]")
		print("psi   :", np.round(psi, 3), "[degrees]")

		# In prototyping fase, run 100 times the for loop and stop afterwards.
		# Also if 5 times no contour (in case FatalError was put to 1) was found directly stop.
		if FinalCountdown > 15 or FatalError == 1:
			break
		else:
			#plt.waitforbuttonpress(0)
			plt.close()
			FinalCountdown = FinalCountdown + 1
			print("##############################")
			print("Iteration:", FinalCountdown)

# Stop when keyboard interrupt is present to ensure safe disconnection with camera
except KeyboardInterrupt:
	pass

finally:
	pipeline.stop()
	print("Camera has succesfully been stopped and disconnected")
	# Timing Analysis
	if DoTimingAnalysis == 1 and FatalError == 0:
		# TIMING:
		time_loading_initials = LoadInitials_timer - Start_timer
		while_loop_timer_append = while_loop_timer_append[1:]
		while_loop_timer_append = while_loop_timer_append[:-1]

		plt.figure()
		plt.rc('font', size=20)  # controls default text sizes
		plt.rc('axes', titlesize=20)  # fontsize of the axes title
		plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
		plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
		plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
		plt.rc('legend', fontsize=16)  # legend fontsize
		plt.rc('figure', titlesize=20)  # fontsize of the figure title

		plt.plot(range(len(while_loop_timer_append)), while_loop_timer_append, linewidth = 3, label="Cycle loop time")
		plt.plot(range(len(while_loop_timer_append)), np.ones(len(while_loop_timer_append)) * np.mean(while_loop_timer_append), linewidth = 3,
				 label="Cycle loop time average: " + str(np.round(np.mean(while_loop_timer_append), 3)) + str(" [s]"))

		plt.legend(loc="upper left")
		plt.title("Cycle loop time")
		plt.xlabel("# of processed images")
		plt.grid()
		plt.ylabel("Time [s]")
		plt.ylim(np.min(while_loop_timer_append) - 0.1 * (np.mean(while_loop_timer_append) - np.min(while_loop_timer_append)), np.max(while_loop_timer_append) - 0.5 * (np.mean(while_loop_timer_append) - np.max(while_loop_timer_append)))
		plt.tight_layout()
		plt.draw()
		plt.waitforbuttonpress(0)
		plt.savefig('timeing.eps', format='eps', dpi=300)

		# Pie chart, where the slices will be ordered and plotted counter-clockwise:
		labels = 'Obtain Image', 'Classify Object', 'Plane Fitting', 'Pose Estimation'
		sizes = [(np.mean(timing_load_images_append) / np.mean(while_loop_timer_append))*100,
				 (np.mean(timing_stop_objectclass_append) / np.mean(while_loop_timer_append)) * 100,
				 (np.mean(timing_stop_depthEstimation_append) / np.mean(while_loop_timer_append)) * 100,
				 (np.mean(timing_stop_findOrient_append) / np.mean(while_loop_timer_append)) * 100]

		explode = (0.05, 0.05, 0.05, 0.05)

		plt.rc('font', size=16)  # controls default text sizes
		plt.rc('axes', titlesize=16)  # fontsize of the axes title
		plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
		plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
		plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
		plt.rc('legend', fontsize=16)  # legend fontsize
		plt.rc('figure', titlesize=16)  # fontsize of the figure title

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
				shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		plt.title("Cycle time distribution")
		plt.draw()
		plt.waitforbuttonpress(0)
		plt.savefig('timeing_pi_chart.eps', format='eps', dpi=300)



