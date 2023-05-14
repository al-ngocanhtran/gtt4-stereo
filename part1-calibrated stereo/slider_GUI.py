import numpy as np 
import cv2


# Check for left and right camera IDs
# These values can change depending on the system
CamR_id = 0 # Camera ID for left camera
CamL_id = 1 # Camera ID for right camera

CamL= cv2.VideoCapture(CamL_id)
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


CamR= cv2.VideoCapture(CamR_id)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("./data/params_py.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

def nothing(x):
    pass

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)

cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
# cv2.createTrackbar('preFilterType','disp',1,1,nothing)
# cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,80,nothing)
# cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',5,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,200,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,150,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,50,nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()

while True:

	# Capturing and storing left and right camera images
	retL, imgL= CamL.read()
	retR, imgR= CamR.read()
	
	# Proceed only if the frames have been captured
	if retL and retR:
		imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

		# Applying stereo image rectification on the left image
		Left_nice= cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
		
		# Applying stereo image rectification on the right image
		Right_nice= cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)

		# Updating the parameters based on the trackbar positions
		numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
		blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
		# preFilterType = cv2.getTrackbarPos('preFilterType','disp')
		# preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
		preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
		# textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
		uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
		speckleRange = cv2.getTrackbarPos('speckleRange','disp')
		speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
		disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
		minDisparity = cv2.getTrackbarPos('minDisparity','disp')
		mode =cv2.STEREO_SGBM_MODE_SGBM_3WAY
		# Setting the updated parameters before computing disparity map
		stereo.setNumDisparities(numDisparities)
		stereo.setBlockSize(blockSize)
		# stereo.setPreFilterType(preFilterType)
		# stereo.setPreFilterSize(preFilterSize)
		stereo.setPreFilterCap(preFilterCap)
		# stereo.setTextureThreshold(textureThreshold)
		stereo.setUniquenessRatio(uniquenessRatio)
		stereo.setSpeckleRange(speckleRange)
		stereo.setSpeckleWindowSize(speckleWindowSize)
		stereo.setDisp12MaxDiff(disp12MaxDiff)
		stereo.setMinDisparity(minDisparity)
		stereo.setMode(mode)


		# Calculating disparity using the StereoBM algorithm
		disparity = stereo.compute(Left_nice,Right_nice)
		# NOTE: compute returns a 16bit signed single channel image,
		# CV_16S containing a disparity map scaled by 16. Hence it 
		# is essential to convert it to CV_32F and scale it down 16 times.

		# Converting to float32 
		disparity = disparity.astype(np.float32)

		# Scaling down the disparity values and normalizing them 
		disparity = (disparity/16.0 - minDisparity)/numDisparities

		# Displaying the disparity map
		cv2.imshow("disp",disparity)

				# Calculating disparity using the StereoBM algorithm
		disparity = stereo.compute(Left_nice,Right_nice)

		# Converting to float32 
		disparity = disparity.astype(np.float32)

		# Scaling down the disparity values and normalizing them 
		disparity = (disparity/16.0 - minDisparity)/numDisparities
		
		max_dist = 230 # max distance to keep the target object (in cm)
		min_dist = 50 # Minimum distance the stereo setup can measure (in cm)
		sample_delta = 40 # Distance between two sampling points (in cm)
		Z = max_dist 

		Value_pairs = []
		y,x = 360, 640
		value_pairs = np.array(Value_pairs)
		z = value_pairs[:,0]
		disp = value_pairs[:,1]
		disp_inv = 1/disp

		if disparity[y,x] > 0:
			Value_pairs.append([Z,disparity[y,x]])
			message = "Distance: {} cm  | Disparity: {}.".format(Z,disparity[y,x])
			cv2.putText(message,(50,50),1,5,(255,0,255),10)
			Z-=sample_delta

		
		# if Z < min_dist:
			
		# Close window using esc key
		if cv2.waitKey(1) == 27:
			break
	
	else:
		CamL= cv2.VideoCapture(CamL_id)
		CamR= cv2.VideoCapture(CamR_id)

print("Saving depth estimation paraeters ......")

cv_file = cv2.FileStorage("./data/disp_params.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("numDisparities",numDisparities)
cv_file.write("blockSize",blockSize)
# cv_file.write("preFilterType",preFilterType)
# cv_file.write("preFilterSize",preFilterSize)
cv_file.write("preFilterCap",preFilterCap)
# cv_file.write("textureThreshold",textureThreshold)
cv_file.write("uniquenessRatio",uniquenessRatio)
cv_file.write("speckleRange",speckleRange)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("disp12MaxDiff",disp12MaxDiff)
cv_file.write("minDisparity",minDisparity)
cv_file.write("M",39.075)
cv_file.release()
