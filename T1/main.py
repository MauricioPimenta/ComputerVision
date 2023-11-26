import sys
from matplotlib import axes, axis
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
from stl import mesh


###### Crie suas funções de translação, rotação, criação de referenciais, plotagem de setas e qualquer outra função que precisar

def getObjFromFile(filename):
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
		self.objeto_original = getObjFromFile(filename) #modificar
		self.objeto = self.objeto_original.copy()
		self.cam_original = np.array([	[1, 0, 0, 0],
										[0, 1, 0, 0],
										[0, 0, 1, 0],
										[0, 0, 0, 1]]) #modificar
		
		self.cam = self.cam_original.copy() #modificar
		self.px_base = 1280  #modificar
		self.px_altura = 720 #modificar
		self.dist_foc = 50 #modificar
		self.stheta = 0 #modificar
		self.ox = self.px_base/2 #modificar
		self.oy = self.px_altura/2 #modificar
		self.ccd = [36,24] #modificar
		self.projection_matrix = [] #modificar


	def setup_ui(self):
		# Criar o layout de grade
		grid_layout = QGridLayout()

		# Criar os widgets
		line_edit_widget1 = self.create_world_widget("Ref mundo")
		line_edit_widget2  = self.create_cam_widget("Ref camera")
		line_edit_widget3  = self.create_intrinsic_widget("params instr")

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

		# Criar um objeto FigureCanvas para exibir o gráfico 2D
		self.fig1, self.ax1 = plt.subplots()
		self.ax1.set_title("Imagem")
		self.canvas1 = FigureCanvas(self.fig1)

		##### Falta acertar os limites do eixo X - grafico 2D
		self.ax1.set_xlim((-self.px_base/2, +self.px_base/2))

		##### Falta acertar os limites do eixo Y - grafico 2D
		self.ax1.set_ylim((-self.px_altura/2, +self.px_altura/2))

		##### Você deverá criar a função de projeção
		object_2d = self.projection_2d()

		##### Falta plotar o object_2d que retornou da projeção
		self.ax1.plot(object_2d[0,:], object_2d[1,:], 'r')


		self.ax1.grid(True)
		self.ax1.set_aspect('equal')
		canvas_layout.addWidget(self.canvas1)

		# Criar um objeto FigureCanvas para exibir o gráfico 3D
		self.fig2 = plt.figure()
		#self.ax2 = plt.Axes
		self.ax2 = self.fig2.add_subplot(111, projection='3d')

		##### Falta plotar o seu objeto 3D e os referenciais da câmera e do mundo

		# Plot the points drawing the lines
		self.ax2.plot(self.objeto_original[0,:],self.objeto_original[1,:],self.objeto_original[2,:],'r')
		self.ax2.set_aspect('equal')


		self.canvas2 = FigureCanvas(self.fig2)
		canvas_layout.addWidget(self.canvas2)

		# Retornar o widget de canvas
		return canvas_widget



	##### Você deverá criar as suas funções aqui

	def update_params_intrinsc(self, line_edits):
		return

	def update_world(self,line_edits):

		return

	def update_cam(self,line_edits):
		return

	def projection_2d(self):

		return

	def generate_intrinsic_params_matrix(self):
		return

	def update_canvas(self):
		return

	def reset_canvas(self):

		return


if __name__ == '__main__':
	app = QApplication(sys.argv)
	main_window = MainWindow()
	main_window.show()
	sys.exit(app.exec_())
