import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHBoxLayout
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import subprocess

from os.path import join, split, exists
from tools.datautils import DataUtils
from tools.debug import Debug

GROUP = "Mindfulness-Project"
dutils = DataUtils()
debug = Debug()

subject_id = sys.argv[1]
session = sys.argv[2]

EXEC_PATH = join(dutils.DEVANALYSEPATH, "graphplot")

class HeatmapWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.similarity_matrix = data["simmatrix_sp"]
        self.labels = data["labels"]
        self.labels_indices = data["labels_indices"]
        self.initUI()

    def initUI(self):
        self.setWindowTitle('2D Heatmap')

        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Create layout for tables and heatmap
        table_layout = QHBoxLayout()
        main_layout.addLayout(table_layout)

        # Create tables
        self.table1 = QTableWidget()
        self.table2 = QTableWidget()
        self.table1.setRowCount(len(self.labels))
        self.table1.setColumnCount(1)
        self.table2.setRowCount(len(self.labels))
        self.table2.setColumnCount(1)
        self.table1.setHorizontalHeaderLabels(["X Parcels"])
        self.table2.setHorizontalHeaderLabels(["Y Parcels"])

        for i, label in enumerate(self.labels):
            self.table1.setItem(i, 0, QTableWidgetItem(label))
            self.table2.setItem(i, 0, QTableWidgetItem(label))

        table_layout.addWidget(self.table1)
        table_layout.addWidget(self.table2)

        self.table1.itemClicked.connect(self.handle_table1_click)
        self.table2.itemClicked.connect(self.handle_table2_click)

        # Create Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        main_layout.addWidget(self.canvas)

        # Plot the heatmap
        self.plot_heatmap()

        # Connect click and motion events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.selected_index1 = None
        self.selected_index2 = None

    def plot_heatmap(self):
        self.ax.clear()
        heatmap = self.ax.imshow(
            self.similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1
        )
        self.figure.colorbar(heatmap, ax=self.ax)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x = int(event.xdata)
        y = int(event.ydata)
        self.ax.set_title(f"Clicked at: ({y}, {x})")
        self.canvas.draw()

        # Execute bash command in background
        self.run_bash_command(x,y)

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            self.ax.set_title("Mouse position: ( , )")
            self.canvas.draw()
            return

        x = int(event.xdata)
        y = int(event.ydata)
        try:
            X_label_id = self.labels_indices[x]
            Y_label_id = self.labels_indices[y]
            parcel_x_string = self.labels[x]
            parcel_y_string = self.labels[y]
            corr = self.similarity_matrix[x, y]
            self.ax.set_title(f"({x}, {y}) \n ({parcel_x_string}, {parcel_y_string}) \n  corr = {round(corr, 2)}")
            self.canvas.draw()
        except Exception:
            pass

    def run_bash_command(self, i, j):
        # Execute the bash command in the background
        try:
            subprocess.Popen(['python3', f'{EXEC_PATH}/plot_correlation.py', subject_id, session, str(i), str(j)])
        except Exception as e:
            print(f"Error running command: {e}")

    def handle_table1_click(self, item):
        self.selected_index1 = item.row()
        self.check_selection()

    def handle_table2_click(self, item):
        self.selected_index2 = item.row()
        self.check_selection()

    def check_selection(self):
        if self.selected_index1 is not None and self.selected_index2 is not None:
            print(f'Selected indices: ({self.selected_index1}, {self.selected_index2})')
            x,y = int(self.selected_index1), int(self.selected_index2)
            self.run_bash_command(x, y)

            # You can perform actions based on the selected indices here

def main():
    dirpath = join(dutils.DATAPATH, GROUP, "derivatives", "connectomes", f"sub-{subject_id}", f"ses-{session}", "spectroscopy")
    filename = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz"

    path = join(dirpath, filename)
    if not exists(path):
        debug.error(f"sub-{subject_id}_ses-{session} not found")
        sys.exit()

    data   = np.load(path)
    app    = QApplication(sys.argv)
    window = HeatmapWindow(data)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
