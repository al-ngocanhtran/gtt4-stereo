import numpy as np
import cv2
import time

print("Checking the right and left camera IDs:")
print("Press (y) if IDs are correct and (n) to swap the IDs")
print("Press enter to start the process >> ")
input()

# Check for left and right camera IDs
CamL_id = 1
CamR_id = 0

CamL= cv2.VideoCapture(CamL_id)
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


CamR= cv2.VideoCapture(CamR_id)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


for i in range(100):
    retL, frameL= CamL.read()
    retR, frameR= CamR.read()

cv2.imshow('imgL',frameL)
cv2.imshow('imgR',frameR)

if cv2.waitKey(0) & 0xFF == ord('y') or cv2.waitKey(0) & 0xFF == ord('Y'):
    CamL_id = 1
    CamR_id = 0
    print("Camera IDs maintained")

elif cv2.waitKey(0) & 0xFF == ord('n') or cv2.waitKey(0) & 0xFF == ord('N'):
    CamL_id = 1
    CamR_id = 0
    print("Camera IDs swapped")
else:
    print("Wrong input response")
    exit(-1)
CamR.release()
CamL.release()

CamL= cv2.VideoCapture(CamL_id)
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


CamR= cv2.VideoCapture(CamR_id)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

output_path = "./data/"
start = time.time()
T = 3
count = 0

while True:
    timer = T - int(time.time() - start)
    retR, frameR= CamR.read()
    retL, frameL= CamL.read()
    
    img1_temp = frameL.copy()
    cv2.putText(img1_temp,"%r"%timer,(50,50),1,5,(255,0,255),10)
    cv2.imshow('imgR',frameR)
    cv2.imshow('imgL',img1_temp)

    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(9,6),None)
    retL, cornersL = cv2.findChessboardCorners(grayL,(9,6),None)

    # If corners are detected in left and right image then we save it.
    if (retR == True) and (retL == True) and timer <=0:
        count+=1
        cv2.imwrite(output_path+'chessR/img%d.jpg'%count,frameR)
        cv2.imwrite(output_path+'chessL/img%d.jpg'%count,frameL)
        print("got %d images:"%count)
    
    if timer <=0:
        start = time.time()
    
    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
