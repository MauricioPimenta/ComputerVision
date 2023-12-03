# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Mauricio Bittencourt Pimenta

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):

	# Get centroid coords (Xc, Yc)
	centroid = np.mean(points, axis=0)

	# translate points to have the centroid in (0,0)
	centered_pts = points - centroid

	# Calculate the average distance of all the points to the centroid (the origin)
	mean_dist = np.mean(np.sqrt(np.sum(np.power(centered_pts, 2), axis=0)))



	norm_points = np.dot(points, T)

	return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):

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
def compute_normalized_dlt(pts1, pts2):

	# # Add homogeneous coordinates
	# pts1 = np.transpose(pts1)
	# pts1 = np.vstack( (pts1, np.ones(np.shape(pts1[0]))) )

	# pts2 = np.transpose(pts2)
	# pts2 = np.vstack( (pts2, np.ones(np.shape(pts2[0]))) )

	# Normaliza pontos
	pts1_norm = normalize_points(pts1)
	pts2_norm = normalize_points(pts2)

	# Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
	A = compute_A(pts1_norm, pts2_norm)

	# Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada

	# Perform SVD(A) = U.S.Vt to estimate the homography
	v = np.linalg.svd(A)    # return all three arrays into 'v'

	# Takes the last column of the last array of 'v' and reshapes it as the homography matrix
	H_matrix = np.reshape(v[-1][-1], (3,3)); print('\nH_Matrix: \n', H_matrix, '\n')

	# Denormaliza H_normalizada e obtém H

	return H_matrix


# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens


def RANSAC(pts1, pts2, dis_threshold, N, Ninl):

	# Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc


	# Processo Iterativo
		# Enquanto não atende a critério de parada

		# Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2

		# Usa as amostras para estimar uma homografia usando o DTL Normalizado

		# Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
		# o número de supostos inliers obtidos com o modelo estimado

		# Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas.
		# Atualiza também o número N de iterações necessárias

	# Terminado o processo iterativo
	# Estima a homografia final H usando todos os inliers selecionados.

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
	M = np.identity(len(src_pts));# AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
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
