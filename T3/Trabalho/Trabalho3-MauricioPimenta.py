'''
 ---- Trabalho 03 - Detecção de posição 3D de um robô usando imagens de 4 câmeras ----


Aluno: Maurício Bittencourt Pimenta

Neste trabalho vocês deverão detectar o robô nos vídeos das 4 câmeras do espaço inteligente e obter a reconstrução da sua posição 3D no mundo. Feito isso, vocês deverão gerar um gráfico da posição do robô, mostrando a trajetória que ele realizou.

Para detectar o robô será usado um marcador ARUCO acoplado a sua plataforma. Rotinas de detecção desse tipo de marcador poderão ser usadas para obter sua posição central, assim como as suas quinas nas imagens. Essas informações, juntamente com os dados de calibração das câmeras, poderão ser usadas para localização 3D do robô.

Informações a serem consideradas:

- Só é necessário a reconstrução do ponto central do robô (ou suas quinas, se vocês acharem melhor). Para isso, vocês podem usar o método explicado no artigo fornecido como material adicional ou nos slides que discutimos em sala de aula.

- O robô está identificado por um marcador do tipo ARUCO - Código ID 0 (zero) - Tamanho 30 x 30 cm

- Os vídeos estão sincronizados para garantir que, a cada quadro, vocês estarão processando imagens do robô capturadas no mesmo instante.

- A calibração das câmeras é fornecida em 4 arquivos no formato JSON (Para saber como ler os dados, basta procurar no Google).

- Rotinas de detecção dos marcadores Aruco em imagens e vídeo são fornecidas logo abaixo.

ATENÇÃO: Existem rotinas de detecção de ARUCO que já fornecem sua localização e orientação 3D, se a calibração da câmera e o tamanho do padrão forem fornecidas. Essas rotinas poderão ser usadas para fazer comparações com a reconstrução 3D fornecida pelo trabalho de vocês, mas não serão aceitas como o trabalho a ser feito. Portanto, lembrem-se que vocês deverão desenvolver a rotina de reconstrução, a partir da detecção do ARUCO acoplado ao robô nas imagens 2D capturadas nos vídeos.


DATA DE ENTREGA: 16/12/2023

'''

# Import libraries
import cv2 as cv
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json  # JSON for handling JSON data

# Define Paths to use
videosPath = ['videos/camera-00.mp4', 'videos/camera-01.mp4', 'videos/camera-02.mp4', 'videos/camera-03.mp4']
cameraDataPath = ["calibration data/0.json", "calibration data/1.json", "calibration data/2.json", "calibration data/3.json"]


# ---------------------------------------------------------------------------------------
#																						|
# 										Functions										|
#																						|
# ---------------------------------------------------------------------------------------

class Camera:

	def __init__(self, intrinsicParametersMatrix, resolution, transformMatrix, lensDistortion):
		self.intrinsic = intrinsicParametersMatrix
		self.resolution = resolution
		self.transform = transformMatrix
		self.distortion = lensDistortion
		self.rotMatrix = transformMatrix[:3, :3]
		self.translationMatrix = transformMatrix[:3, 3].reshape(3, 1)
		self.projectionMatrix = Camera.computeProjectionMatrix(self.intrinsic, self.rotMatrix, self.translationMatrix)
	
	
	def getIntrinsic(self):
		return self.intrinsic

	def getResolution(self):
		return self.resolution

	def getTransform(self):
		return self.transform

	def getDistortion(self):
		return self.distortion

	# Function to read the intrinsic and extrinsic parameters of each camera
	def getCameraParametersFromFile(file):

		# Load the JSON data
		camera_data = json.load(open(file))

		# Parse the intrinsic matrix, resolution, transformation matrix, and distortion coefficients
		intrinsicMatrix = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)

		resolution = [camera_data['resolution']['width'], camera_data['resolution']['height']]

		transformMatrix = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)

		lensDistortion = np.array(camera_data['distortion']['doubles'])

		# Return a camera object with the intrinsic matrix, resolution, transformation matrix, and distortion coefficients

		Cam = Camera(intrinsicMatrix, resolution, transformMatrix, lensDistortion)

		return Cam

	def computeProjectionMatrix(K, R, T):
		# The projection matrix is calculated as K[R|T], where R is the rotation matrix and T is the translation vector.
		# Here, instead of directly multiplying the matrices, we first append [0,0,0,1] at the bottom of the concatenated R and T matrices to make it a 4x4 matrix.
		# Then, we apply the inverse of this 4x4 matrix, get the first three rows (to get back to a 3x4 matrix) and finally multiply with the intrinsic matrix K.
		# This operation essentially performs the multiplication K*[inv([R|T])], but in the homogeneous coordinate system.
		return np.dot(K, np.linalg.inv(np.vstack((np.hstack((R, T)),np.array([0,0,0,1]))))[:-1,:])

def loadVideos(videosPath):

	# List to store detected points
	points = list()

	for video in videosPath:

		# Open the video file - Create a VideoCapture object with the video file
		videoFile = cv.VideoCapture(video)

		# Check if the video file was opened successfully
		if not videoFile.isOpened():
			print("\n\nError opening video file\n\n")

		# Temporary list to store points detected in each frame
		tmpoints = list()

		# Read the video frame by frame
		while videoFile.isOpened():
			# Read a frame from the video
			ret, frame = videoFile.read()

			# If a frame was successfully read:
			if not ret:
				print("\nCan't receive frame (video end?). Exiting ...\n")
				break
			else:
				# Convert the frame to grayscale
				gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

				# Detect the markers in the frame
				corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

				# If markers were detected, compute the average corner points
				if corners:
					c_aux = np.array(corners[0]) # Takes only the first marker if it finds more than one
												# The Aruco that we are looking for is of id 0 so it will always be the first
					tmpoints.append(np.mean(c_aux[0], axis=0))
				else:
					tmpoints.append(np.array([None,None]))  # If no markers were detected, append None values

				# Draw the detected markers on the frame and display the marked frame
				frame_marked = aruco.drawDetectedMarkers(frame, corners, ids)
				cv.imshow('frame', frame_marked)

				# If the user presses 'q', break the loop
				if cv.waitKey(1) == ord('q'):
					break

		# Append the detected points from this video to the main points list
		points.append(tmpoints)

		# When everything done, release the video file
		videoFile.release()

		# Close all OpenCV windows
		cv.destroyAllWindows()

	return points



# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
#																						|
# 										Main Code										|
#																						|
# ---------------------------------------------------------------------------------------

# Load the predefined dictionary for Aruco marker detection from OpenCV
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters()

points = loadVideos(videosPath)

# Convert points list to numpy array and replace None values with infinity
mean_points = np.array(points)
mean_points[mean_points == None] = np.inf

# Load the camera parameters from the JSON files
Cam0 = Camera.getCameraParametersFromFile(cameraDataPath[0])
Cam1 = Camera.getCameraParametersFromFile(cameraDataPath[1])
Cam2 = Camera.getCameraParametersFromFile(cameraDataPath[2])
Cam3 = Camera.getCameraParametersFromFile(cameraDataPath[3])

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
Points_3d = list()

# For each frame, calculate the 3D points
for i in range(len(points_camera0)):
	# Prepare matrices and perform Singular Value Decomposition (SVD) for each camera
	# to compute the 3D coordinates, using a linear triangulation method.
	# Insert the computed 3D coordinates into the Points_3d list
	# For each camera, if the point is valid, calculate its corresponding matrix; otherwise, assign a zero matrix

	# For camera 0
	if points_camera0[i][0] != np.inf:
		C0 = np.hstack((Cam0.projectionMatrix, -np.array(points_camera0[i]).reshape(-1, 1).astype(float)))
		C0 = np.hstack((C0, np.zeros((3, 1))))
		C0 = np.hstack((C0, np.zeros((3, 1))))
		C0 = np.hstack((C0, np.zeros((3, 1))))
	else:
		C0 = np.zeros((1, 8), dtype=np.float64)

	# For camera 1
	if points_camera1[i][0] != np.inf:
		C1 = np.hstack((Cam1.projectionMatrix, np.zeros((3, 1))))
		C1 = np.hstack((C1, -np.array(points_camera1[i]).reshape(-1, 1).astype(float)))
		C1 = np.hstack((C1, np.zeros((3, 1))))
		C1 = np.hstack((C1, np.zeros((3, 1))))
	else:
		C1 = np.zeros((1, 8), dtype=np.float64)

	# For camera 2
	if points_camera2[i][0] != np.inf:
		C2 = np.hstack((Cam2.projectionMatrix, np.zeros((3, 1))))
		C2 = np.hstack((C2, np.zeros((3, 1))))
		C2 = np.hstack((C2, -np.array(points_camera2[i]).reshape(-1, 1).astype(float)))
		C2 = np.hstack((C2, np.zeros((3, 1))))
	else:
		C2 = np.zeros((1, 8), dtype=np.float64)

	# For camera 3
	if points_camera3[i][0] != np.inf:
		C3 = np.hstack((Cam3.projectionMatrix, np.zeros((3, 1))))
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
