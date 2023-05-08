import numpy as np 
import cv2
import matplotlib.pyplot as plt

pathR = "./data/boxR.jpg"
pathL = "./data/boxL.jpg" 

cv_file = cv2.FileStorage("./data/params_py.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

stereo = cv2.StereoSGBM_create()

imgL= cv2.imread(pathL)
imgR= cv2.imread(pathR)

imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

Left_nice= cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
		
# Applying stereo image rectification on the right image
Right_nice= cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)

# grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
# grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

plt.figure(figsize = (20, 10))
plt.subplot(121); plt.imshow(np.hstack([imgL, imgR]), cmap='gray'); plt.title('Normal');
plt.subplot(122); plt.imshow(np.hstack([Left_nice, Right_nice]), cmap='gray'); plt.title('Rectified');
plt.savefig("hoa.png")

disparity = stereo.compute(Left_nice,Right_nice)
disparity = disparity.astype(np.float32)
minDisparity = 5
numDisparities = 1
disparity = (disparity/16.0 - minDisparity)/numDisparities
cv2.imwrite("hoa_disp.jpg",disparity)

