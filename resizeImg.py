import cv2
import numpy as np 

path = "./data/imgL.jpg"
img = cv2.imread(path)
img_new = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
cv2.imwrite(path, img_new)


path = "./data/imgR.jpg"
img = cv2.imread(path)
img_new = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
cv2.imwrite(path, img_new)




