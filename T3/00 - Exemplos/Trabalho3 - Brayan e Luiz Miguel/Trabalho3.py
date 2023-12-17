'''
By Brayan Moreno Arevalo and Luiz Miguel Monteiro Nascimento Pessotti Tavares

Course: Computer Vision (Prof. Raquel Frizera Vassallo)
Institution: UFES
Date: July/2023

Title - 3D Marker Detection and Reconstruction from Multiple Cameras

Language and Libraries:

    Programming Language: Python - 3.10.6
    Libraries:
        OpenCv - 4.8.0
        Matplotlib - 3.7.1
        Numpy - 1.24.4
        Json - 2.0.9
'''

# Import the necessary libraries
import cv2  # OpenCV for image processing and computer vision tasks
import cv2.aruco as aruco  # ArUco for marker detection
import matplotlib.pyplot as plt  # matplotlib for plotting data
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import numpy as np  # NumPy for array manipulation
import json  # JSON for handling JSON data

# Define the paths of the video files to be processed
videos = ['videos/camera-00.mp4', 'videos/camera-01.mp4', 'videos/camera-02.mp4', 'videos/camera-03.mp4']

# Load the predefined dictionary for ArUco marker detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters()

# Define lists to store the detected marker points and ids
points = []
for video in videos:

    # Open the video file
    cap = cv2.VideoCapture(video)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    
    # Temporary list to store points detected in each frame
    aux = []

    # Read the video frame by frame
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        
        # If a frame was successfully read:
        if ret:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the markers in the frame
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)
            
            # If markers were detected, compute the average corner points
            if corners:
                c_aux = np.array(corners[0]) # Takes only the first marker if it finds more than one
                                             # The Aruco that we are looking for is of id 0 so it will always be the first 
                aux.append(np.mean(c_aux[0], axis=0))
            else:
                aux.append(np.array([None,None]))  # If no markers were detected, append None values
                
            # Draw the detected markers on the frame
            frame_marked = aruco.drawDetectedMarkers(frame, corners, ids)

            # Display the marked frame
            cv2.imshow('frame', frame_marked)

            # If the user presses 'q', break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Append the detected points from this video to the main points list
    points.append(aux)

    # Release the video file
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Convert points list to numpy array and replace None values with infinity
mean_points = np.array(points)
mean_points[mean_points == None] = np.inf

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    # Load the JSON data
    camera_data = json.load(open(file))
    # Parse the intrinsic matrix, resolution, transformation matrix, and distortion coefficients
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'], camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

# Load the camera parameters from the JSON files
K0, R0, T0, res0, dis0 = camera_parameters('json/0.json')
K1, R1, T1, res1, dis1 = camera_parameters('json/1.json')
K2, R2, T2, res2, dis2 = camera_parameters('json/2.json')
K3, R3, T3, res3, dis3 = camera_parameters('json/3.json')

def proj_matrix(K, R, T):
    # The projection matrix is calculated as K[R|T], where R is the rotation matrix and T is the translation vector.
    # Here, instead of directly multiplying the matrices, we first append [0,0,0,1] at the bottom of the concatenated R and T matrices to make it a 4x4 matrix.
    # Then, we apply the inverse of this 4x4 matrix, get the first three rows (to get back to a 3x4 matrix) and finally multiply with the intrinsic matrix K.
    # This operation essentially performs the multiplication K*[inv([R|T])], but in the homogeneous coordinate system.
    return np.dot(K, np.linalg.inv(np.vstack((np.hstack((R, T)),np.array([0,0,0,1]))))[:-1,:])

# Calculate the projection matrix for each camera
P0 = proj_matrix(K0, R0, T0)
P1 = proj_matrix(K1, R1, T1)
P2 = proj_matrix(K2, R2, T2)
P3 = proj_matrix(K3, R3, T3)

# Get the detected points from each camera
points_camera0 = mean_points[0]
points_camera1 = mean_points[1]
points_camera2 = mean_points[2]
points_camera3 = mean_points[3]

# Add a column of ones to the detected points, transforming them into homogeneous coordinates
points_camera0 = np.hstack((points_camera0, np.ones((len(points_camera0), 1))))
points_camera1 = np.hstack((points_camera1, np.ones((len(points_camera1), 1))))
points_camera2 = np.hstack((points_camera2, np.ones((len(points_camera2), 1))))
points_camera3 = np.hstack((points_camera3, np.ones((len(points_camera3), 1))))

# Create an empty list to store the 3D points
Points_3d = []

# For each frame, calculate the 3D points
for i in range(len(points_camera0)):
    # Prepare matrices and perform Singular Value Decomposition (SVD) for each camera
    # to compute the 3D coordinates, using a linear triangulation method.
    # Insert the computed 3D coordinates into the Points_3d list
    # For each camera, if the point is valid, calculate its corresponding matrix; otherwise, assign a zero matrix

    # For camera 0
    if points_camera0[i][0] != np.inf:
        C0 = np.hstack((P0, -np.array(points_camera0[i]).reshape(-1, 1).astype(float)))
        C0 = np.hstack((C0, np.zeros((3, 1))))
        C0 = np.hstack((C0, np.zeros((3, 1))))
        C0 = np.hstack((C0, np.zeros((3, 1))))
    else:
        C0 = np.zeros((1, 8), dtype=np.float64)

    # For camera 1
    if points_camera1[i][0] != np.inf:
        C1 = np.hstack((P1, np.zeros((3, 1))))
        C1 = np.hstack((C1, -np.array(points_camera1[i]).reshape(-1, 1).astype(float)))
        C1 = np.hstack((C1, np.zeros((3, 1))))
        C1 = np.hstack((C1, np.zeros((3, 1))))
    else:
        C1 = np.zeros((1, 8), dtype=np.float64)

    # For camera 2
    if points_camera2[i][0] != np.inf:
        C2 = np.hstack((P2, np.zeros((3, 1))))
        C2 = np.hstack((C2, np.zeros((3, 1))))
        C2 = np.hstack((C2, -np.array(points_camera2[i]).reshape(-1, 1).astype(float)))
        C2 = np.hstack((C2, np.zeros((3, 1))))
    else:
        C2 = np.zeros((1, 8), dtype=np.float64)

    # For camera 3
    if points_camera3[i][0] != np.inf:
        C3 = np.hstack((P3, np.zeros((3, 1))))
        C3 = np.hstack((C3, np.zeros((3, 1))))
        C3 = np.hstack((C3, np.zeros((3, 1))))
        C3 = np.hstack((C3, -np.array(points_camera3[i]).reshape(-1, 1).astype(float)))
    else:
        C3 = np.zeros((1, 8), dtype=np.float64)

    # Stack the matrices from all cameras into one large matrix
    B_matrix = np.vstack((C0, C1, C2, C3))

    # Remove the rows and columns of zeros
    mask = np.any(B_matrix != 0, axis=1) 
    mask2 = np.any(B_matrix != 0, axis=0)
    B_matrix_n = B_matrix[mask]
    B_matrix_n = B_matrix_n[:,mask2]

    # Use SVD to solve the system of linear equations to get the 3D point
    _, _, D = np.linalg.svd(B_matrix_n)

    # Append the 3D point to the list
    Points_3d.append(D[-1][:4])

# Convert the list of 3D points into a numpy array and make them homogeneous
Points_3d = np.array(Points_3d)
Points_3d = Points_3d / Points_3d[:, 3].reshape(-1, 1)

# Extract the X, Y, Z coordinates
X = Points_3d[:, 0]
Y = Points_3d[:, 1]
Z = Points_3d[:, 2]

# Filter the points where Z is greater than -1
X = X[np.where(Z > -1)]
Y = Y[np.where(Z > -1)]
Z = Z[np.where(Z > -1)]

# Create a new 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.plot(X, Y, Z)

# Set the labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the limits for the axes
ax.set_xlim([-1.8,1.8])
ax.set_ylim([-1.8,1.8])
ax.set_zlim([0,3.6])

# Create new figures for each coordinate in 2D
fig = plt.figure()
plt.plot(X)
plt.title('X')

fig = plt.figure()
plt.plot(Y)
plt.title('Y')

fig = plt.figure()
plt.plot(Z)
plt.title('Z')
plt.ylim([0.5, 0.7])

# Show all the plots
plt.show()