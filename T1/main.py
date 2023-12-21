import sys
from matplotlib import axes, axis
from matplotlib.pylab import Axes
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
from stl import mesh
from math import pi,cos,sin


###### Crie suas funções de translação, rotação, criação de referenciais, plotagem de setas e qualquer outra função que precisar

### Funções auxiliares -------------------------------------------

def move (dx,dy,dz):
	T = np.eye(4)
	T[0,-1] = dx
	T[1,-1] = dy
	T[2,-1] = dz
	return T

def z_rotation(angle):
	rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
	return rotation_matrix

def x_rotation(angle):
	rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
	return rotation_matrix

def y_rotation(angle):
	rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
	return rotation_matrix


def set_plot(ax: Axes3D = None, figure = None, lim = [-2,2]):
	if figure ==None:
		figure = plt.figure(figsize=(8,8))
	if ax==None:
		ax = plt.axes(projection='3d')

	ax.set_title("camera referecnce")
	ax.set_xlim(lim)
	ax.set_xlabel("x axis")
	ax.set_ylim(lim)
	ax.set_ylabel("y axis")
	ax.set_zlim(lim)
	ax.set_zlabel("z axis")
	return ax

#adding quivers to the plot
def draw_arrows(point, base, axis, length=1.5):
	# The object base is a matrix, where each column represents the vector
	# of one of the axis, written in homogeneous coordinates (ax,ay,az,0)


	# Plot vector of x-axis
	axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
	# Plot vector of y-axis
	axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
	# Plot vector of z-axis
	axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)

	return axis

def getObjFromFile(filename) -> np.ndarray :
	# Load the STL files and add the vectors to the plot
	your_mesh = mesh.Mesh.from_file(filename)

	# Get the x, y, z coordinates contained in the mesh structure that are the
	# vertices of the triangular faces of the object
	x = your_mesh.x.flatten()
	y = your_mesh.y.flatten()
	z = your_mesh.z.flatten()

	# Get the vectors that define the triangular faces that form the 3D object
	object = your_mesh.vectors

	# Create the 3D object from the x,y,z coordinates and add the additional array of ones to
	# represent the object using homogeneous coordinates
	return np.array([x.T,y.T,z.T,np.ones(x.size)])



class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		# definindo as variaveis
		self.set_variables()

  		#Ajustando a tela
		self.setWindowTitle("Grid Layout")
		self.setGeometry(100, 100,1280 , 720)
		self.setup_ui()

	def set_variables(self):
		filename = 'megaman.STL'

		self.objeto_original = getObjFromFile(filename) # Read Object from STL file
		self.objeto = self.objeto_original.copy()		# Saves a copy of the object to work with

		self.cam_original = np.array([	[1, 0, 0, 0],
										[0, 1, 0, 0],
										[0, 0, 1, 0],
										[0, 0, 0, 1]])
		self.cam = self.cam_original.copy()

		self.ccd = [36,24]		# Tamanho do sensor da camera, em mm (36 x 24)

		self.px_base = 1280		# largura da imagem em px
		self.px_altura = 720	# altura da imagem em px
		self.img = np.array([self.px_base, self.px_altura]) # tamanho da imagem gerada, em px (1280 x 720)

		self.f = 50		# Distancia Focal, em mm

		# Fatores de Escala
		self.Sx = self.img[0]/self.ccd[0]	# Sx = n_pixels_base/ccd_x
		self.Sy = self.img[1]/self.ccd[1]	# Sy = n_pixels_altura/ccd_y
		self.Stheta = 0						# S theta - Skew = 0 nesse caso

		# Pontos da Referencia
		# Ox e Oy sao as coordenadas, em pixels, do ponto central da imagem
		self.Ox = self.px_base/2
		self.Oy = self.px_altura/2

		self.intrinsicParamMatrix = np.array([	[self.f*self.Sx,	self.Stheta,		self.Ox	],
												[ 		0, 			self.f*self.Sy,		self.Oy	],
												[ 		0, 				0,					1	]])

		self.extrinsicParamMatrix = np.eye(4)

		self.projection_matrix = [] #modificar

		self.WorldRef = np.eye(4);	# Referencial do Mundo

	def setup_ui(self):
		# Criar o layout de grade
		grid_layout = QGridLayout()

		# Criar os widgets
		line_edit_widget1 = self.create_world_widget("Ref mundo")
		line_edit_widget2  = self.create_cam_widget("Ref camera")
		line_edit_widget3  = self.create_intrinsic_widget("params intrins")

		self.canvas = self.create_matplotlib_canvas()

		# Adicionar os widgets ao layout de grade
		grid_layout.addWidget(line_edit_widget1, 0, 0)
		grid_layout.addWidget(line_edit_widget2, 0, 1)
		grid_layout.addWidget(line_edit_widget3, 0, 2)
		grid_layout.addWidget(self.canvas, 1, 0, 1, 3)

		# Criar um widget para agrupar o botão de reset
		reset_widget = QWidget()
		reset_layout = QHBoxLayout()
		reset_widget.setLayout(reset_layout)

		# Criar o botão de reset vermelho
		reset_button = QPushButton("Reset")
		reset_button.setFixedSize(50, 30)  # Define um tamanho fixo para o botão (largura: 50 pixels, altura: 30 pixels)
		style_sheet = """
			QPushButton {
				color : white ;
				background: rgba(255, 127, 130,128);
				font: inherit;
				border-radius: 5px;
				line-height: 1;
			}
		"""
		reset_button.setStyleSheet(style_sheet)
		reset_button.clicked.connect(self.reset_canvas)

		# Adicionar o botão de reset ao layout
		reset_layout.addWidget(reset_button)

		# Adicionar o widget de reset ao layout de grade
		grid_layout.addWidget(reset_widget, 2, 0, 1, 3)

		# Criar um widget central e definir o layout de grade como seu layout
		central_widget = QWidget()
		central_widget.setLayout(grid_layout)

		# Definir o widget central na janela principal
		self.setCentralWidget(central_widget)

	def create_intrinsic_widget(self, title):
		# Criar um widget para agrupar os QLineEdit
		line_edit_widget = QGroupBox(title)
		line_edit_layout = QVBoxLayout()
		line_edit_widget.setLayout(line_edit_layout)

		# Criar um layout de grade para dividir os QLineEdit em 3 colunas
		grid_layout = QGridLayout()

		line_edits = []
		labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']  # Texto a ser exibido antes de cada QLineEdit

		# Adicionar widgets QLineEdit com caixa de texto ao layout de grade
		for i in range(1, 7):
			line_edit = QLineEdit()
			label = QLabel(labels[i-1])
			validator = QDoubleValidator()  # Validador numérico
			line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
			grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
			grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
			line_edits.append(line_edit)

		# Criar o botão de atualização
		update_button = QPushButton("Atualizar")

		##### Você deverá criar, no espaço reservado ao final, a função self.update_params_intrinsc ou outra que você queira
		# Conectar a função de atualização aos sinais de clique do botão
		update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

		# Adicionar os widgets ao layout do widget line_edit_widget
		line_edit_layout.addLayout(grid_layout)
		line_edit_layout.addWidget(update_button)

		# Retornar o widget e a lista de caixas de texto
		return line_edit_widget

	def create_world_widget(self, title):
		# Criar um widget para agrupar os QLineEdit
		line_edit_widget = QGroupBox(title)
		line_edit_layout = QVBoxLayout()
		line_edit_widget.setLayout(line_edit_layout)

		# Criar um layout de grade para dividir os QLineEdit em 3 colunas
		grid_layout = QGridLayout()

		line_edits = []
		labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

		# Adicionar widgets QLineEdit com caixa de texto ao layout de grade
		for i in range(1, 7):
			line_edit = QLineEdit()
			label = QLabel(labels[i-1])
			validator = QDoubleValidator()  # Validador numérico
			line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
			grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
			grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
			line_edits.append(line_edit)

		# Criar o botão de atualização
		update_button = QPushButton("Atualizar")

		##### Você deverá criar, no espaço reservado ao final, a função self.update_world ou outra que você queira
		# Conectar a função de atualização aos sinais de clique do botão
		update_button.clicked.connect(lambda: self.update_world(line_edits))

		# Adicionar os widgets ao layout do widget line_edit_widget
		line_edit_layout.addLayout(grid_layout)
		line_edit_layout.addWidget(update_button)

		# Retornar o widget e a lista de caixas de texto
		return line_edit_widget

	def create_cam_widget(self, title):
	 # Criar um widget para agrupar os QLineEdit
		line_edit_widget = QGroupBox(title)
		line_edit_layout = QVBoxLayout()
		line_edit_widget.setLayout(line_edit_layout)

		# Criar um layout de grade para dividir os QLineEdit em 3 colunas
		grid_layout = QGridLayout()

		line_edits = []
		labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

		# Adicionar widgets QLineEdit com caixa de texto ao layout de grade
		for i in range(1, 7):
			line_edit = QLineEdit()
			label = QLabel(labels[i-1])
			validator = QDoubleValidator()  # Validador numérico
			line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
			grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
			grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
			line_edits.append(line_edit)

		# Criar o botão de atualização
		update_button = QPushButton("Atualizar")

		##### Você deverá criar, no espaço reservado ao final, a função self.update_cam ou outra que você queira
		# Conectar a função de atualização aos sinais de clique do botão
		update_button.clicked.connect(lambda: self.update_cam(line_edits))

		# Adicionar os widgets ao layout do widget line_edit_widget
		line_edit_layout.addLayout(grid_layout)
		line_edit_layout.addWidget(update_button)

		# Retornar o widget e a lista de caixas de texto
		return line_edit_widget

	def create_matplotlib_canvas(self):
		# Criar um widget para exibir os gráficos do Matplotlib
		canvas_widget = QWidget()
		canvas_layout = QHBoxLayout()
		canvas_widget.setLayout(canvas_layout)

		# initialize fig1 and ax2D
		self.fig2D, self.ax2D = plt.subplots()

		# initialize fig3D and ax3D
		self.fig3D = plt.figure()
		self.ax3D: Axes = self.fig3D.add_subplot(111, projection='3d')

		# Create the FigureCanvas Objects to display the 2D and 3D plots
		self.canvas2D = FigureCanvas(self.fig2D)
		canvas_layout.addWidget(self.canvas2D)

		self.canvas3D = FigureCanvas(self.fig3D)
		canvas_layout.addWidget(self.canvas3D)

		# Call the method to update the plots and display them in the canvas
		self.update_canvas()

		# Retornar o widget de canvas
		return canvas_widget



	##### Você deverá criar as suas funções aqui

	def update_params_intrinsc(self, line_edits):
		# line_edits contains the list of QLineEdit widgets in the followinf order:
		# [n_pixels_base, n_pixels_altura, ccd_x, ccd_y, dist_focal, sθ]
		# each line_edits contains a double floating point number

		# Check if the line_edits are empty before converting them to float
		# and updating the intrinsic parameters
		if line_edits[0].text() != '':
			self.px_base = float(line_edits[0].text())
			print('\n', self.px_base ,'\n')

		if line_edits[1].text() != '':
			self.px_altura = float(line_edits[1].text())
			print('\n', self.px_altura ,'\n')

		if line_edits[2].text() != '':
			self.ccd[0] = float(line_edits[2].text())
			print('\n', self.ccd[0] ,'\n')

		if line_edits[3].text() != '':
			self.ccd[1] = float(line_edits[3].text())
			print('\n', self.ccd[1] ,'\n')

		if line_edits[4].text() != '':
			self.f = float(line_edits[4].text())
			print('\n', self.f ,'\n')

		if line_edits[5].text() != '':
			self.Stheta = float(line_edits[5].text())	# S theta - Skew = 0 nesse caso
			print('\n', self.Stheta ,'\n')


		# Update the intrinsic parameters
		self.img = np.array([self.px_base, self.px_altura]) # tamanho da imagem gerada, em px (1280 x 720)

		# Fatores de Escala
		self.Sx = self.img[0]/self.ccd[0]	# Sx = n_pixels_base/ccd_x
		self.Sy = self.img[1]/self.ccd[1]	# Sy = n_pixels_altura/ccd_y

		# Pontos da Referencia
		# Ox e Oy sao as coordenadas, em pixels, do ponto central da imagem
		self.Ox = self.px_base/2
		self.Oy = self.px_altura/2

		self.generate_intrinsic_params_matrix()

		self.update_canvas()
		return

	def update_world(self,line_edits):

		return

	def update_cam(self,line_edits):
		'''
			This Function updates the camera matrix based on the values of the line_edits.
			It moves the camera by the ammount specified in the 'line_edits' always in accordance
			  to the camera reference system. If the camera is moved in the x axis, it will move in
			  its own x axis, and so on..

			-> line_edits contains the list of QLineEdit widgets in the following order:
			    [X(move), X(angle), Y(move), Y(angle), Z(move), Z(angle)]

			Each line_edits contains a double floating point number
		'''
		# Temporary Matrix to store the transformations
		TransformMatrix = np.eye(4)

		if line_edits[0].text() != '':
			dx = float(line_edits[0].text())
			TransformMatrix = np.dot(TransformMatrix, move(dx,0,0))
			print('\n dx: ', dx ,'\n')

		if line_edits[1].text() != '':
			theta_x = float(line_edits[1].text())
			TransformMatrix = np.dot(TransformMatrix, x_rotation(np.deg2rad(theta_x)))
			print('\n angX: ', theta_x ,'\n')

		if line_edits[2].text() != '':
			dy = float(line_edits[2].text())
			TransformMatrix = np.dot(TransformMatrix, move(0,dy,0))
			print('\n dy: ', dy ,'\n')

		if line_edits[3].text() != '':
			theta_y = float(line_edits[3].text())
			TransformMatrix = np.dot(TransformMatrix, y_rotation(np.deg2rad(theta_y)))
			print('\n angY: ', theta_y ,'\n')

		if line_edits[4].text() != '':
			dz = float(line_edits[4].text())
			TransformMatrix = np.dot(TransformMatrix, move(0,0,dz))
			print('\n dz: ', dz ,'\n')

		if line_edits[5].text() != '':
			# Get the angle in degrees
			theta_z = float(line_edits[5].text())

			# Update the TransformMatrix
			TransformMatrix = np.dot(TransformMatrix, z_rotation(np.deg2rad(theta_z)))
			print('\n angZ: ', theta_z ,'\n')


		# Update the camera matrix using the transformation matrixes
		print('\n Tmat: \n', TransformMatrix, '\n')

		self.cam = np.dot(TransformMatrix, self.cam)
		self.extrinsicParamMatrix = self.cam.copy()
		print('\n Extrinsic: ', self.extrinsicParamMatrix, '\n')

		self.update_canvas()

		return

	def projection_2d(self):
		# Projectio Matrix = IntrinsicMatrix x PIo x ExtrinsicMatrix
		PIo = np.eye(3,4, dtype='float')

		Proj = np.linalg.multi_dot([self.intrinsicParamMatrix, PIo, self.extrinsicParamMatrix])

		print('\n ProjM: ', Proj, '\n')

		# Projection and creation of the image
		object_2d = np.dot(Proj, self.objeto)

		# Preparacao das coordenadas na forma cartesiana
		object_2d /= object_2d[-1]		# Normaliza a coord 'z'
		print('\nObj2D: ', object_2d, '\n')

		self.projection_matrix = Proj

		return object_2d

	def generate_intrinsic_params_matrix(self):
		self.intrinsicParamMatrix = np.array([	[self.f*self.Sx,	self.Stheta,		self.Ox	],
												[ 		0, 			self.f*self.Sy,		self.Oy	],
												[ 		0, 				0,					1	]])
		return

	# Update the canvas with the plots of the 2D and 3D objects
	def update_canvas(self):

		## Clear and update the 2D Plot and Canvas
		self.ax2D.clear()

		self.ax2D.set_title("Imagem")
		##### Falta acertar os limites do eixo X - grafico 2D
		self.ax2D.set_xlim((0, self.px_base))

		##### Falta acertar os limites do eixo Y - grafico 2D
		self.ax2D.set_ylim(self.px_altura, 0)

		##### Você deverá criar a função de projeção
		object_2d = self.projection_2d()

		##### Falta plotar o object_2d que retornou da projeção
		self.ax2D.plot(object_2d[0], object_2d[1], 'g')

		self.ax2D.grid(True)
		self.ax2D.set_aspect('equal')

		self.canvas2D.draw()


		## Clear and update the 3D Plot and Canvas
		self.ax3D.clear()

		# Plot the points drawing the lines
		self.ax3D.plot(self.objeto_original[0,:],self.objeto_original[1,:],self.objeto_original[2,:],'r')
		# self.ax3D.set_xlim(-40,40)
		# self.ax3D.set_ylim(-40,40)
		self.ax3D.set_aspect('equal')

		# Draw Camera Arrows
		self.ax3D = draw_arrows(self.cam[:,-1], self.cam[:,0:3], self.ax3D, 10)

		# Draw World Reference Arrows
		self.ax3D = draw_arrows(self.WorldRef[:,-1], self.WorldRef[:,0:3], self.ax3D, 10)

		self.canvas3D.draw()

		return


	def reset_canvas(self):
		self.set_variables()
		self.update_canvas()

		return



if __name__ == '__main__':
	app = QApplication(sys.argv)
	main_window = MainWindow()
	main_window.show()
	sys.exit(app.exec_())
