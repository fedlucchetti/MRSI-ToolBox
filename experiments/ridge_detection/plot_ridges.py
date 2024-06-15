import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt
from tools.datautils import DataUtils
dutils    = DataUtils()

class ImageAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Analysis with Sliders")
        self.setGeometry(100, 100, 800, 600)

        # Create the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Generate a sample 2D array (image data)
        data_list     = dutils.load_all_orig_res()
        img_list,_,_ = data_list
        imgdata       = img_list[0]
        print(imgdata.shape)
        # imgdata,header = dutils.load_nii(file_type="OrigResEigenA",fileid="A009",metabolic_str="ACr+PCr",normalization=True,rawnii=False)
        self.data,_,_ = dutils.normalize_image(imgdata)
        print(self.data.shape)
        self.data  = self.data[:,20,:]
        # self.data = np.random.rand(100, 100) * 255
        # self.data = self.data.astype(np.uint8)

        # Create the image plot
        self.img_view = pg.ImageView()
        self.img_view.setImage(self.data)
        layout.addWidget(self.img_view)

        # Create the histogram plot
        self.histogram = pg.PlotWidget()
        y, x = np.histogram(self.data, bins=np.arange(257))
        self.histogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        layout.addWidget(self.histogram)

        # Create the lower and upper bound sliders
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setRange(0, 255)
        self.lower_slider.setValue(0)
        self.lower_slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.lower_slider)

        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setRange(0, 255)
        self.upper_slider.setValue(255)
        self.upper_slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.upper_slider)

    def update_image(self):
        # Get the slider values
        lower = self.lower_slider.value()
        upper = self.upper_slider.value()

        # Update the image based on the slider values
        mask = (self.data >= lower) & (self.data <= upper)
        filtered_data = np.where(mask, self.data, 0)
        self.img_view.setImage(filtered_data)

def main():
    app = QApplication(sys.argv)
    main_window = ImageAnalysisApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
