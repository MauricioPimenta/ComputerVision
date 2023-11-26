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
	if len(points.shape) == 1:
		points = points[:, np.newaxis]

	# Calculate centroid
	centroid = np.mean(points, axis=0)
	# Calculate the average distance of the points having the centroid as origin
	mean_distance = np.mean(np.sqrt(np.sum((points - centroid[:, None])**2, axis=0)))

	if mean_distance == 0:
		return points, np.eye(3)

	# Define the scale to have the average distance as sqrt(2)
	
	# Define the normalization matrix (similar transformation)
	
	# Normalize points
	



	T = np.array([[np.sqrt(2)/mean_distance, 0, -np.sqrt(2)/mean_distance * centroid[0]],
				  [0, np.sqrt(2)/mean_distance, -np.sqrt(2)/mean_distance * centroid[1]],
				  [0, 0, 1]])

	norm_points = np.dot(T[:2, :2], points) + T[:2, 2][:, None]


	return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):

	A = np.zeros((2 * len(pts1), 9))

	for i in range(len(pts1[0])):
		x, y = pts1[:, i]
		xp, yp = pts2[:, i]
		A[2 * i, :] = [-x, -y, 0, 0, 0, 0, x * xp, y * xp, xp]
		A[2 * i + 1, :] = [0, 0, 0, -x, -y, 0, x * yp, y * yp, yp]

	return A


# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):

	# Normaliza pontos
	norm_pts1, T1 = normalize_points(pts1)
	norm_pts2, T2 = normalize_points(pts2)

	# Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
	A = compute_A(norm_pts1, norm_pts2)

	# Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada
	try:
		_, _, V = np.linalg.svd(A)
		H_normalized = V[-1, :].reshape(3, 3)

		# Denormaliza H_normalizada e obtém H
		H = np.dot(np.linalg.inv(T2), np.dot(H_normalized, T1))

		return H

	except np.linalg.LinAlgError:
		print("A matriz de homografia não é invertível.")
		return np.eye(3)


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
	max_inliers = 0
	best_H = None
	best_pts1_in = None
	best_pts2_in = None

	numPoints = len(pts1); print('\n\n' , numPoints , '\n\n')
	minPoints = 4

	#pts1 = np.transpose(pts1)[0]
	#pts2 = np.transpose(pts2)[0]

	# Processo Iterativo
	for _ in range(int(N)):
		# Enquanto não atende a critério de parada
		if len(pts1[0]) >= 4:

			# Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2
			indices = np.random.choice(numPoints, size=minPoints, replace=False)
			sampled_pts1 = pts1[indices]
			sampled_pts2 = pts2[indices]

			# Usa as amostras para estimar uma homografia usando o DTL Normalizado
			H = compute_normalized_dlt(sampled_pts1, sampled_pts2)

			# Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
			# o número de supostos inliers obtidos com o modelo estimado
			distances = np.sqrt(np.sum((pts2 - np.dot(H, pts1))**2, axis=0))
			inliers = np.sum(distances < dis_threshold)

			# Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas.
			# Atualiza também o número N de iterações necessárias
			if inliers > max_inliers:
				max_inliers = inliers
				best_H = H
				best_pts1_in = pts1[distances < dis_threshold]
				best_pts2_in = pts2[distances < dis_threshold]

				p_inlier = inliers / len(pts1[0])
				epsilon = 1e-10
				N = min(N, np.log(epsilon) / np.log(1 - p_inlier**4))

	# Terminado o processo iterativo
	# Estima a homografia final H usando todos os inliers selecionados.
	H_final = compute_normalized_dlt(best_pts1_in, best_pts2_in)

	return H_final, best_pts1_in, best_pts2_in


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

# Verifica se há pontos de correspondência suficientes
if len(kp1) < MIN_MATCH_COUNT or len(kp2) < MIN_MATCH_COUNT:
	print("Not enough matches are found - {}/{}".format(min(len(kp1), len(kp2)), MIN_MATCH_COUNT))
	exit()

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
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

	# Aplicação da função RANSAC
	#################################################
	M, pts1_in, pts2_in = RANSAC(src_pts, dst_pts, dis_threshold=2, N=1000, Ninl=MIN_MATCH_COUNT)   # AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
	#################################################

	# Aplicação da homografia na imagem de entrada
	img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

else:
	print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
	matchesMask = None

# Desenho dos matches e visualização das imagens
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