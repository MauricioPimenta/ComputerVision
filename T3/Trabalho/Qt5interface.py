import sys
import matplotlib as mpl
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		
		self.setWindowTitle("Trabalho 3")
		self.setGeometry(100, 100, 800, 600)

		
		self.main_widget = QWidget(self)
		self.setCentralWidget(self.main_widget)
		self.layout = QGridLayout(self.main_widget)
		self.main_widget.setLayout(self.layout)
		self.plot_widget = PlotCanvas(self.main_widget)
		self.layout.addWidget(self.plot_widget, 0, 0, 1, 2)
		self.input_widget = InputWidget(self.main_widget)
		self.layout.addWidget(self.input_widget, 1, 0, 1, 2)
		self.input_widget.plot_button.clicked.connect(self.plot_widget.plot)
		self.show()