import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys




file_name = "video01.mkv" 

parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)


vid = cv2.VideoCapture(file_name)


while True:
    _, img = vid.read()
    if img is None:
        print("Empty Frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(img)
    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow('output', frame_markers)
    
    
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

