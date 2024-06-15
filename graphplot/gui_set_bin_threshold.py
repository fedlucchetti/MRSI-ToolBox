import sys
import numpy as np
import copy
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QTableWidget, QTableWidgetItem, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from tools.datautils import DataUtils
from os.path import join, split, exists
from bids.mridata import MRIData
from tools.filetools import FileTools
from tools.debug import Debug
import csv

dutils = DataUtils()
GROUP  = "Mindfulness-Project"
ftools = FileTools(GROUP)
debug  = Debug()
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, simmatrix_binarized, degrees, degree_counts, X_fit, y_pred_huber,ransac):

        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        r_squared = ransac.score(degrees.reshape(-1, 1), np.log(degree_counts))
        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].imshow(simmatrix_binarized, aspect='auto', cmap='magma')
        self.axes[0].set_title('Binarized Similarity Matrix')

        self.axes[1].plot(degrees, degree_counts, ".", color='r', alpha=0.7)
        self.axes[1].plot(X_fit, np.exp(y_pred_huber), color='red', label=f"slope {round(slope,2)}---R2:{round(r_squared,2)}")
        self.axes[1].set_xlabel('Degree', fontsize=12)
        self.axes[1].set_ylabel('Counts', fontsize=12)
        self.axes[1].set_xlim(0,80)
        self.axes[1].set_ylim(1,100)
        self.axes[1].set_yscale('log')
        self.axes[1].legend()
        self.axes[1].grid()
        self.axes[1].set_title('Degree Distribution')

        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQt Plotting Example')
        self.setGeometry(100, 100, 1400, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QHBoxLayout(self.main_widget)

        self.plot_canvas = PlotCanvas(self.main_widget)
        layout.addWidget(self.plot_canvas)

        control_layout = QVBoxLayout()

        self.slider = QSlider(Qt.Vertical, self)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.update_plots)
        control_layout.addWidget(self.slider)

        self.slider_label = QLabel(f'Threshold: {self.slider.value() / 1000.0:.3f}', self)
        control_layout.addWidget(self.slider_label)

        self.save_button = QPushButton('Save Threshold', self)
        self.save_button.clicked.connect(self.save_threshold)
        control_layout.addWidget(self.save_button)

        control_layout.setAlignment(Qt.AlignTop)
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setFixedWidth(125)
        layout.addWidget(control_widget)

        recording_list = np.array(ftools.list_recordings())
        self.recording_list = recording_list[np.argsort(recording_list[:, 0]), :]
        self.threshold_list = np.zeros(len(self.recording_list))
        
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(len(self.recording_list))
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(['Recording', 'Version', 'Threshold'])
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.horizontalHeader().setStretchLastSection(False)
        self.table_widget.setColumnWidth(0, 80)
        self.table_widget.setColumnWidth(1, 50)
        self.table_widget.setColumnWidth(2, 80)

        # Set fixed width for the table
        table_width = self.table_widget.columnWidth(0) + self.table_widget.columnWidth(1) + self.table_widget.columnWidth(2) + self.table_widget.verticalHeader().width() + 4
        self.table_widget.setFixedWidth(table_width)

        for i, (recording, version) in enumerate(self.recording_list):
            self.table_widget.setItem(i, 0, QTableWidgetItem(recording))
            self.table_widget.setItem(i, 1, QTableWidgetItem(version))
            self.table_widget.setItem(i, 2, QTableWidgetItem(str(self.threshold_list[i])))

        self.load_thresholds()

        self.table_widget.cellClicked.connect(self.on_table_item_clicked)
        layout.addWidget(self.table_widget)

        # Init Data and Plots
        self.subject_id, self.session = self.recording_list[0]
        self.update_simmatrix()
        self.update_plots()

    # def degree_distribution(self,simmatrix,threshold):
    #     simmatrix_adjusted  = copy.deepcopy(simmatrix)
    #     simmatrix_binarized = copy.deepcopy(simmatrix_adjusted)
    #     simmatrix_binarized[np.abs(simmatrix_adjusted) < threshold] = 0
    #     simmatrix_binarized[np.abs(simmatrix_adjusted) >= threshold] = np.sign(simmatrix_adjusted[np.abs(simmatrix_adjusted) >= threshold])

    #     degree_distribution = self.degree_distribution(simmatrix_binarized)
    #     degrees = np.array(list(degree_distribution.keys()))
    #     ids = np.argsort(degrees)
    #     degree_counts = np.array(list(degree_distribution.values()))
    #     degrees, degree_counts = degrees[ids[1::]], degree_counts[ids[1::]]

    #     # Get parameters from the RANSAC model
    #     ransac = RANSACRegressor()
    #     ransac.fit(degrees.reshape(-1, 1), np.log(degree_counts))
    #     X_fit        = np.linspace(degrees.min(), degrees.max(), 100).reshape(-1, 1)
    #     y_pred_huber = ransac.predict(X_fit)
    #     return simmatrix_binarized, degrees, degree_counts, X_fit, y_pred_huber,ransac

    def update_plots(self):
        try:
            threshold = self.slider.value() / 1000.0
            self.slider_label.setText(f'Threshold: {threshold:.3f}')
            simmatrix_adjusted = copy.deepcopy(self.simmatrix)
            simmatrix_binarized = copy.deepcopy(simmatrix_adjusted)
            simmatrix_binarized[np.abs(simmatrix_adjusted) < threshold] = 0
            simmatrix_binarized[np.abs(simmatrix_adjusted) >= threshold] = np.sign(simmatrix_adjusted[np.abs(simmatrix_adjusted) >= threshold])

            degree_distribution = self.degree_distribution(simmatrix_binarized)
            degrees = np.array(list(degree_distribution.keys()))
            ids = np.argsort(degrees)
            degree_counts = np.array(list(degree_distribution.values()))
            degrees, degree_counts = degrees[ids[1::]], degree_counts[ids[1::]]

            # Get parameters from the RANSAC model
            ransac = RANSACRegressor()
            ransac.fit(degrees.reshape(-1, 1), np.log(degree_counts))
            X_fit = np.linspace(degrees.min(), degrees.max(), 100).reshape(-1, 1)
            y_pred_huber = ransac.predict(X_fit)
            # simmatrix_binarized, degrees, degree_counts, X_fit, y_pred_huber,ransac = self.degree_distribution(simmatrix,threshold)
            self.plot_canvas.plot(simmatrix_binarized, degrees, degree_counts, X_fit, y_pred_huber,ransac)
        except Exception as e:
            debug.warning(e)

    def save_threshold(self):
        threshold = self.slider.value() / 1000.0
        for i, (recording, version) in enumerate(self.recording_list):
            if recording == self.subject_id and version == self.session:
                self.threshold_list[i] = threshold
                self.table_widget.setItem(i, 2, QTableWidgetItem(str(threshold)))
                break

        file_path = join(dutils.ANARESULTSPATH, "thresholds.csv")
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Recording", "Version", "Threshold"])
            for i, (recording, version) in enumerate(self.recording_list):
                writer.writerow([recording, version, self.threshold_list[i]])
        print(f"Thresholds saved to {file_path}")

    def load_thresholds(self):
        file_path = join(dutils.ANARESULTSPATH, "thresholds.csv")
        if exists(file_path):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    recording, version, threshold = row
                    for i, (rec, ver) in enumerate(self.recording_list):
                        if rec == recording and ver == version:
                            self.threshold_list[i] = float(threshold)
                            self.table_widget.setItem(i, 2, QTableWidgetItem(threshold))
                            break
            print(f"Thresholds loaded from {file_path}")

    def update_simmatrix(self):
        # Load Data
        mridata = MRIData(self.subject_id, self.session,group=GROUP)
        connectome_dir_path = f"{mridata.ROOT_PATH}/derivatives/connectomes/sub-{self.subject_id}/ses-{self.session}/spectroscopy"
        anat_parcel_nifti = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
        anat_parcel_mrsi_nifti = anat_parcel_nifti.replace("space-orig", "space-mrsi")
        outfilename = anat_parcel_mrsi_nifti.split('/')[-1].replace("space-mrsi_", "").replace(".nii.gz", "_simmatrix.npz")
        simmatrix_path = f"{connectome_dir_path}/{outfilename}"
        self.simmatrix = np.load(simmatrix_path)["simmatrix_mi"]
        # pvalues   = np.load(simmatrix_path)["pvalue_mi"]
        # self.simmatrix[pvalues>0.0001] = 0
        debug.info("Simmatrix loaded", self.simmatrix.shape)

    def degree_distribution(self, simmatrix_binarized):
        degrees = np.sum(simmatrix_binarized != 0, axis=1)
        unique, counts = np.unique(degrees, return_counts=True)
        return dict(zip(unique, counts))

    def on_table_item_clicked(self, row, column):
        self.subject_id = self.recording_list[row, 0]
        self.session    = self.recording_list[row, 1]
        # self.threshold  = self.threshold_list[row]
        # self.slider.value = self.threshold_list[row]
        # self.slider.sliderPosition = self.threshold_list[row]
        # self.slider.update()
        self.update_simmatrix()
        debug.success(f"Selected Recording: {self.subject_id}, Session: {self.session}")
        self.update_plots()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())






