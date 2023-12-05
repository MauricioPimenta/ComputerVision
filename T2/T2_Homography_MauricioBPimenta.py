# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Mauricio Bittencourt Pimenta

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math as m
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points : np.ndarray) -> [np.ndarray , np.ndarray] :

	"""
	Nomalize_points:
	----------------

	Function to normalize points.\n
	It returns a ``ndarray`` containing the normalized points in homogeneous coordinates and 
	the Transformation Matrix ``T`` to change the ``points`` to the normalized space.
	\n\n

	``norm_points = T @ points``

	Parameters:
	-----------
		``points`` : ``numpy.ndarray``
			``array`` with coordinates of the points (x,y) from the image to be normalized.
			This array is expected not to be in homogeneous coordinates.

	Returns:
	--------
		``norm_points``: ``numpy.ndarray``
			array containing the normalized points in homogeneous coordinates.\n
			| xi |\n
			| yi |\n
			| wi ]\n

		``T`` : ``numpy.ndarray``
			Transformation Matrix to get the normalized points
	"""

	if points.ndim != 2 :
		raise ValueError("\n normalize_points: Wrong number of dimensions for array of points.\n",
				   		 "expected: 2\n",
						 "Array provided has ",points.ndim," dimensions")
	if len(points) < 4 :
		raise ValueError("\n normalize_points: Array of points has fewer sets of points than necessary.\n",
				   		 "expected: 4\n",
						 "Array provided has ",len(points)," sets of points")

	# Get centroid coords (Xc, Yc)
	centroid = np.mean(points, axis=0)

	# translate points to have the centroid in (0,0)
	centered_pts = points - centroid

	# Calculate the average distance of all the points to the centroid (the origin)
	mean_dist = np.mean(np.sqrt(np.sum(np.power(centered_pts, 2), axis=0)))

	# Define the scale so the average distance is sqrt(2)
	scale = np.sqrt(2)/mean_dist

	# Define the Transformation Matrix to Normalize the points
	T = np.array([[ scale  ,  0  , -scale*centroid[0]],
			      [   0  , scale , -scale*centroid[1]],
				  [   0  ,   0   ,        1          ]])

	# change points to the homogeneous coordinates
	points_h = np.row_stack((points.T, np.ones(len(points))))
	norm_points = T @ points_h

	return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1 : np.ndarray, pts2 : np.ndarray) -> np.ndarray :

	"""
	compute_A:
	----------

	Function to build the matrix A that defines the system of equations for the DLT.\n
	This function assumes that ``pts1`` and ``pts2`` are in homogeneous coordinates with the last row as w - the homogeneous scale parameter.

	Parameters:
	-----------
		``pts1`` : ``numpy.ndarray``
			``array`` with coordinates of the points (x,y) from the first image.

		``pts2`` : ``numpy.ndarray``
			``array`` with coordinates of the points (x,y) from the second image.

	Returns:
	--------
		``A`` : ``numpy.ndarray``
			Matrix A build from pts1 and pts2

	Notes:
	------

	For every corresponding point in pts1 (x1, y1, w1) and pts2 (x2, y2, w2), the matrix A is calculated as:

	[ 00.00 , 00.00 , 00.00 , -w2*x1 , -w2*y1 , -w2*w1 , y2*x1 , y2*y1 , y2*w1 ] ,\n
	[ w2*x1 , w2*y1 , w2*w1 , 00.00 , 00.00 , 00.00 , -x2*x1 , -x2*y1 , -x2*w1 ]

	"""

	# Compute matrix A
	# Inicializa a Matriz A com zeros -> Tamanho: 2*(qtd de pontos (x,y) de pts1) de altura e 9 de largura (colunas)
	A = np.zeros((2*pts1.shape[1], 9))

	# Para um ponto, a matriz A é calculada como:
	#
	# |     0         0         0       -wi2*xi1   -wi2*yi1   -wi2*wi1   yi2*xi1    yi2*yi1    yi2*wi1   |
	# |  wi2*xi1   wi2*yi1   wi2*wi1        0          0          0     -xi2*xi1   -xi2*yi1   -xi2*wi1   |
	#
	for i in range(pts1.shape[1]):

		# Calcula a primeira linha da matriz A pra todos os pontos
		A[2*i, 3:6] = -pts2[2, i]*pts1[:,i]     # A[3:6] = -wi2*Xi.T
		A[2*i, 6:9] = +pts2[1, i]*pts1[:,i]     # A[6:9] = +yi2*Xi.T

		# Calcula a segunda linha da matriz A pra todos os pontos
		A[2*i+1, 0:3] = +pts2[2, i]*pts1[:,i]   # A[0:3] = +wi2*Xi.T
		A[2*i+1, 6:9] = -pts2[0, i]*pts1[:,i]   # A[6:9] = -xi2*Xi.T


	return A

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1 : np.ndarray, pts2 : np.ndarray) -> np.ndarray :

	"""
	Compute_normalized_dlt:
	-----------------------

	Function to compute the normalized DLT from two corresponding points in two images.\n
	Computes and returns the estimated homography matrix ``H`` from two sets of points, ``pts1`` and ``pts2``.\n
	``pts1`` and ``pts2`` must have at least 4 points to be able to compute ``H``.

	Parameters:
	-----------
		``pts1`` : numpy.ndarray
			points from the first image

		``pts2`` : numpy.ndarray
			points from the second image

	Returns:
	--------
		``H_matrix`` : numpy.ndarray
			Computed Homography Matrix

	"""

	# Normaliza pontos
	pts1_norm, T1 = normalize_points(pts1)
	pts2_norm, T2 = normalize_points(pts2)

	# Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
	A = compute_A(pts1_norm, pts2_norm)

	# Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada

	# Perform SVD(A) = U.S.Vt to estimate the homography
	v = np.linalg.svd(A)    # return all three arrays into 'v'

	# Takes the last column of the last array of 'v' and reshapes it as the homography matrix
	H_norm = np.reshape(v[-1][-1], (3,3)); print('\nH_Matrix: \n', H_norm, '\n')

	# Denormaliza H_normalizada e obtém H
	# we know that { Xĩ' = Hn.Xĩ }  (1)
	# |Xĩ = T1.Xi	(2)
	# |Xĩ' = T2.Xi' (3)
	#
	# So replacing (3) and (2) in (1), we'll have:
	# T2.Xi' = Hn.T1.Xi   ->   Xi' = inv(T2).Hn.T1.Xi
	# H = inv(T2)*H_norm*T1
	H_matrix = (np.linalg.inv(T2)).dot(H_norm.dot(T1))

	return H_matrix


# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
#
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens
def RANSAC(pts1 : np.ndarray, pts2 : np.ndarray, dis_threshold : float = 10.0, N : int = 10000, Ninl : int = 0) -> np.ndarray :

	"""
	RANSAC
	------
	Robust estimation of homography using RANSAC to remove Outliers from data.

	-> Outliers are points with non gaussian error distribution (different and possibly unmodelled error distribution).

	Parameters:
	-----------
		``pts1`` : ``numpy.ndarray`` with 2 dimensions.
			points from first image

		``pts2`` : ``numpy.ndarray`` with 2 dimensions.
			points from second image

		``dis_threshold`` : ``int``, default : 10
			distance threshold for RANSAC - the maximum distance for points to be included as inliers.

		``N`` : ``int``, default : 10000
			Maximum number of iterations allowed in the algorithm - change dinamically based on number of inliers/outliers

		``Ninl`` : ``int``, default : ``0 - zero``
			Desired number of inliers. Defines the number of close data points (inliers) required to assert that the model fits well to the data. - zero by default

	Returns
	-------
		``H`` : ``numpy.ndarray``
			Estimated Homography Matrix if found. ``None`` otherwise.

		``pts1_in`` : ``numpy.ndarray``
			inliers from the first image

		``pts2_in`` : ``numpy.ndarray``
			inliers from the second image

	"""
	# Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc

	# maximum data array dimension - defined as 2 because it is an image (x,y)
	arrayDimensions = 2

	# minimum number of data points required to estimate the model parameters.
	minPoints = 4

	# number of data points of the array passed as parameter to the function
	numPoints = len(pts1)

	# number of iterations so far - will change with number of inliers detected during the algorithm
	iterations = N

	# probability to compute the new number of iterations
	prob = 0.99

	# store the highest number of inliers so far - Start with the required number of inliers
	max_inliers_count = Ninl

	# Best homography matrix so far, computed from the inliers
	best_H : np.ndarray = []

	# the best data set of inliers so far
	best_pts1_in : np.ndarray = []
	best_pts2_in : np.ndarray = []



	# Initial tests..
	# check correct number of dimensions and minimun number of points
	if pts1.ndim > arrayDimensions :
		print("\n\nRANSAC FUNCTION: \n",
		"pts1 has greater number of dimensions than expected.\n",
		"Expected: ",arrayDimensions,".\n",
		"pts1 has: ", pts1.ndim,
		"\n\n")
		return
	elif len(pts1) < minPoints :
		print("\n\nRANSAC FUNCTION: \n",
		"pts1 should have more than 4 points to compute the RANSAC\n",
		"Expected: ",minPoints,".\n",
		"pts1 has: ", len(pts1),
		"\n\n")
		return
	if pts2.ndim > arrayDimensions :
		print("\n\nRANSAC FUNCTION: \n",
		"pts2 has greater number of dimensions than expected.\n",
		"Expected: ",arrayDimensions,".\n",
		"pts2 has: ", pts2.ndim,
		"\n\n")
		return
	elif len(pts2) < minPoints :
		print("\n\nRANSAC FUNCTION: \n",
		"pts2 should have more than 4 points to compute the RANSAC\n",
		"Expected: ",minPoints,".\n",
		"pts2 has: ", len(pts2),
		"\n\n")
		return

	# Check number of inliers passed as parameter


	# Processo Iterativo
	for i in range(iterations) :
		# Enquanto não atende a critério de parada

		# Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2
		indices = np.random.choice(numPoints, size=minPoints, replace=False)
		sampled_pts1 = pts1[indices]
		sampled_pts2 = pts2[indices]

		# Usa as amostras para estimar uma homografia usando o DTL Normalizado
		H_sampled = compute_normalized_dlt(sampled_pts1, sampled_pts2)

		# Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
		# o número de supostos inliers obtidos com o modelo estimado
		inliersCount = 0

		# vectors to store the inliers
		pts1_inliers = np.array(([[]]))
		pts2_inliers = np.array(([[]]))

		for j in range(numPoints) :
			# check result of homography for pts1
			pts2_1 = H_sampled @ (np.append(pts1[j], 1))
			pts2_1 /= pts2_1[2]	# divide by w
			dist_pts1 = np.linalg.norm(pts2_1[0:2] - pts2[j])

			pts1_2 = np.linalg.inv(H_sampled) @ (np.append(pts2[j], 1))
			pts1_2 /= pts1_2[2]
			dist_pts2 = np.linalg.norm(pts1_2[0:2] - pts1[j])

			dist = dist_pts1 + dist_pts2

			if dist < dis_threshold :
				inliersCount += 1
				#pts1_inliers = np.append(np.atleast_2d(pts1_inliers), np.atleast_2d(np.copy(pts1[j])))
				pts1_inliers = np.append(np.atleast_2d(pts1_inliers), np.atleast_2d(pts1[j]), axis=0)
				pts2_inliers = np.append(pts2_inliers, np.atleast_2d(pts2[j]), axis=0)

				pts1_inliers = np.array(np.ndarray, dtype = 'float32', )

		# Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas.
		# Atualiza também o número N de iterações necessárias
		if inliersCount > max_inliers_count :
			max_inliers_count = inliersCount	# get new number of inliers
			best_pts1_in = np.pts1_inliers
			best_pts2_in = pts2_inliers
			best_H = compute_normalized_dlt(best_pts1_in, best_pts2_in)

			# Compute new number of necessary iterations
			w = (max_inliers_count/numPoints)
			k = np.log10(1 - prob) / np.log10(1 - w**minPoints)
			if k < iterations : iterations = k


	# Terminado o processo iterativo
	if best_H == None :
		raise ValueError('\n\n-- RANSAC -- No model found for these sets of points \n\n')

	# Estima a homografia final H usando todos os inliers selecionados.
	H = best_H
	pts1_in = best_pts1_in
	pts2_in = best_pts2_in

	return H, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('./imagens/comicsStarWars01.jpg', 0)   # queryImage
img2 = cv.imread('./imagens/comicsStarWars02.jpg', 0)        # trainImage

# Inicialização do SIFT
try:
	# Tentar usar a versão mais recente
	sift = cv.SIFT_create()
except AttributeError:
	# Se falhar, tentar a versão mais antiga
	sift = cv.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
	if m.distance < 0.75 * n.distance:
		good.append(m)

if len(good) > MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])#.reshape(-1, 1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])#.reshape(-1, 1, 2)

	#################################################
	M = RANSAC(src_pts, dst_pts);	# AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
	#################################################

	img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

else:
	print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
	matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
