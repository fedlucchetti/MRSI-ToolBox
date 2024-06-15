import numpy as np
import matplotlib as plt
import math, sys, os
import nibabel as nib

 
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QFileDialog, QPushButton, QStyle, QSpinBox, QSplitter
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont



from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

from vispy import scene
from vispy.visuals.transforms import STTransform



import pyqtgraph as pg
from pyqtgraph import InfiniteLine

from tools.datautils import DataUtils
from tools.debug import Debug

# from tools.ClickableIMG import ClickableImageView 

utils = DataUtils()
debug = Debug()

# DATAPATH = "/Users/flucchetti/Documents/Connectome/Data/MRSI_reconstructed/Basic"



def create_custom_colormap():
    # Create an array of colors (256 colors, RGBA format)
    colors = np.ones((256, 4), dtype=np.uint8) * 255
    colors[:, 0] = np.arange(256)  # Red channel (grayscale)
    colors[:, 1] = np.arange(256)  # Green channel (grayscale)
    colors[:, 2] = np.arange(256)  # Blue channel (grayscale)
    colors[0, :] = [255, 0, 0, 255]  # Set the first color (0) as red

    # Create and return a ColorMap object
    colormap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=colors)
    return colormap.getLookupTable()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.DATAPATH = os.path.join(utils.DATAPATH,"MRSI_reconstructed")
        self.file_types = ["Qmask", "Conc", "Basic", "Holes","OrigRes",
                           "BasicEigenA","BasicEigenB","BasicEigenC"]
        self.metabolic_current = "Cr+PCr"
        self.metabolics_labels = ["Cr", "Glx", "GPC", "Ins", "NAA"]  # Class attribute for labels

        self.selectedPixels = set()  # Keep track of selected pixels
        data, header = utils.load_nii(file_type="Holes",fileid=1)
        self.unique_ids = utils.list_unique_ids(self.file_types)  # Assuming this method exists and returns a list of IDs
        print(self.unique_ids)
        data = np.flip(data, axis=2)
        self.tensor3D_current=data
        self.current_axis = 0  # 0 for x-axis, 1 for y-axis, 2 for z-axis
        self.custom_colormap = create_custom_colormap()
        self.initUI()

    def initUI(self):
        # Init positions for sldiers
        self.init_pos = np.array([self.tensor3D_current.shape[0]/2,self.tensor3D_current.shape[1]/2,self.tensor3D_current.shape[2]/2]).astype(int)

        ##################### LHS Figure #####################
        self.imageView1 = pg.ImageView(self)
        self.imageView1.getImageItem().mousePressEvent = self.imageClicked
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.slider1.setValue(self.init_pos[0])
        self.spinBox1 = QSpinBox(self)
        self.spinBox1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.spinBox1.setValue(self.init_pos[0])
        self.spinBox1.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox1.setFixedWidth(self.spinBox1.sizeHint().width() + 20)  # Adjust widtg
         # Add  lines to the first image
        self.vLine1 = InfiniteLine(angle=90, movable=False, pen='g')  
        self.imageView1.addItem(self.vLine1)
        self.hLine1 = InfiniteLine(angle=0, movable=False, pen='g') 
        self.imageView1.addItem(self.hLine1)



        ##################### Middle Figure #####################
        self.imageView2 = pg.ImageView(self)
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.slider2.setValue(self.init_pos[1])
        self.spinBox2 = QSpinBox(self)
        self.spinBox2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.spinBox2.setValue(self.init_pos[1])
        self.spinBox2.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox2.setFixedWidth(self.spinBox2.sizeHint().width() + 20)  # Adjust width
        # Add  lines to the second image
        self.hLine2 = InfiniteLine(angle=0, movable=False, pen='g')  # Horizontal line, green
        self.imageView2.addItem(self.hLine2)
        self.vLine2 = InfiniteLine(angle=90, movable=False, pen='g')  # Horizontal line, green
        self.imageView2.addItem(self.vLine2)



        ##################### RHS Figure #####################
        self.imageView3 = pg.ImageView(self)
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.slider3.setValue(self.init_pos[2])
        self.spinBox3 = QSpinBox(self)
        self.spinBox3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.spinBox3.setValue(self.init_pos[2])
        self.spinBox3.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox3.setFixedWidth(self.spinBox2.sizeHint().width() + 20)  # Adjust width
        # Add  lines to the third image
        self.hLine3= InfiniteLine(angle=0, movable=False, pen='g')  # Horizontal line, green
        self.imageView3.addItem(self.hLine3)
        self.vLine3 = InfiniteLine(angle=90, movable=False, pen='g')  # Horizontal line, green
        self.imageView3.addItem(self.vLine3)



        #################### TABLE ####################
        vLayoutTable = QVBoxLayout()
        self.idTable = QTableWidget(self)
        self.idTable.setColumnCount(1)
        self.idTable.setHorizontalHeaderLabels(["Patient IDs"])
        self.populate_id_table()
        self.idTable.cellClicked.connect(self.table_select)
        vLayoutTable.addWidget(self.idTable)
        
        #################### 3D rendering ####################
        vispy_canvas_widget = self.create_vispy_canvas()
        #################### Metabolite SLider ####################
        sliderMetaboliteLayout = self.create_metabolite_slider()
        ####################  XYZ Plane Buttons ####################
        buttonXYZPlaneLayout   = self.create_3D_plane_buttonss()
        

        ##################### Connectors #####################
        self.slider1.valueChanged[int].connect(lambda value: self.updateFromSlider1(value))
        self.spinBox1.valueChanged[int].connect(self.updateFromSpinBox1)
        self.slider2.valueChanged[int].connect(lambda value: self.updateFromSlider2(value))
        self.spinBox2.valueChanged[int].connect(self.updateFromSpinBox2)
        self.slider3.valueChanged[int].connect(lambda value: self.updateFromSlider3(value))
        self.spinBox3.valueChanged[int].connect(self.updateFromSpinBox3)

        self.slider1.valueChanged.connect(self.on_slider_value_changed)
        self.slider2.valueChanged.connect(self.on_slider_value_changed)
        self.slider3.valueChanged.connect(self.on_slider_value_changed)


        ##################### Layout ALL #####################
        vLayout1 = QVBoxLayout()
        vLayout1.addWidget(self.imageView1)
        vLayout1.addWidget(self.slider1)
        vLayout1.addWidget(self.spinBox1)
        vLayout2 = QVBoxLayout()
        vLayout2.addWidget(self.imageView2)
        vLayout2.addWidget(self.slider2)
        vLayout2.addWidget(self.spinBox2)
        vLayout3 = QVBoxLayout()
        vLayout3.addWidget(self.imageView3)
        vLayout3.addWidget(self.slider3)
        vLayout3.addWidget(self.spinBox3)



        # Horizontal layout for the group of vLayout1, vLayout2, vLayout3
        hLayoutMain = QHBoxLayout()
        hLayoutMain.addLayout(vLayout1)
        hLayoutMain.addLayout(vLayout2)
        hLayoutMain.addLayout(vLayout3)

        # Widget for the horizontal layout
        hLayoutWidget = QWidget()
        hLayoutWidget.setLayout(hLayoutMain)

        # Widget for the combined layout of slider and Vispy canvas
        canvasSliderWidget = QWidget()
        hLayoutCanvasSlider = QHBoxLayout(canvasSliderWidget)
        hLayoutCanvasSlider.addLayout(sliderMetaboliteLayout)
        hLayoutCanvasSlider.addLayout(buttonXYZPlaneLayout)
        hLayoutCanvasSlider.addWidget(vispy_canvas_widget)


        # Replace vispy_canvas_widget with the new horizontal layout in verticalSplitter
        verticalSplitter = QSplitter(Qt.Vertical)
        verticalSplitter.addWidget(hLayoutWidget)
        verticalSplitter.addWidget(canvasSliderWidget)  # Add the combined layout widget




        # Widget for plot layout
        plotWidget = QWidget()
        plotLayout = QVBoxLayout(plotWidget)
        plotLayout.addWidget(verticalSplitter)

        # Widget for the table layout
        tableWidget = QWidget()
        tableLayout = QVBoxLayout(tableWidget)
        tableLayout.addLayout(vLayoutTable)

        # Create a horizontal splitter and add the table and plot layouts
        horizontalSplitter = QSplitter(Qt.Horizontal)
        horizontalSplitter.addWidget(tableWidget)
        horizontalSplitter.addWidget(plotWidget)

        # Set the main layout
        mainWidget = QWidget()
        mainWidgetLayout = QVBoxLayout(mainWidget)
        mainWidgetLayout.addWidget(horizontalSplitter)
        self.setCentralWidget(mainWidget)

        # self.plot3DWindow = Plot3DWindow(self.tensor3D_current)
        # self.plot3DWindow.show()
        

         ##################### First Update #####################
        self.hLine1.setPos(self.slider3.value())
        self.vLine1.setPos(self.slider2.value())
        self.hLine2.setPos(self.slider3.value())
        self.vLine2.setPos(self.slider1.value())
        self.hLine3.setPos(self.slider2.value())
        self.vLine3.setPos(self.slider1.value())
        self.updateImage(self.imageView1, self.init_pos[0], axis=0)
        self.updateImage(self.imageView2, self.init_pos[1], axis=1)
        self.updateImage(self.imageView3, self.init_pos[2], axis=2)

    #    # Button to open file explorer
    #     self.loadButton = QPushButton(self)
    #     self.loadButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
    #     self.loadButton.clicked.connect(self.openFileNameDialog)
    #     self.loadButton.move(10, 40)  # Adjust position as needed (e.g., lower)
    #     self.loadButton.show()

    def populate_id_table(self):
        unique_ids_info = self.unique_ids
        header_list = ["ID", "Qmask", "Conc","Basic", "Holes","BasicEigenA","BasicEigenB","BasicEigenC"]
        self.idTable.setColumnCount(len(header_list))
        self.idTable.setHorizontalHeaderLabels(header_list)
        self.idTable.setRowCount(len(unique_ids_info))

        for row, (id, file_info) in enumerate(unique_ids_info.items()):
            self.idTable.setItem(row, 0, QTableWidgetItem(str(id)))
            for col, file_type in enumerate(header_list[1::], start=1):
                self.idTable.setItem(row, col, QTableWidgetItem('Yes' if file_info[file_type] else 'No'))

    def create_metabolite_slider(self):
        # Create a vertical slider
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(4)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setValue(0)  # Default value

        self.slider.valueChanged.connect(self.on_metabolic_slider_value_changed)

        # Create a layout for the slider and labels
        sliderLayout = QVBoxLayout()
        sliderLayout.addWidget(self.slider)

        # Labels for the slider positions
        labels = ["Cr", "Glx", "GPC", "Ins", "NAA"]
        labelLayout = QVBoxLayout()

        # Custom font for the labels
        font = QFont()
        font.setPointSize(16)  # Adjust the font size as needed

        for label_name in labels:
            lbl = QLabel(label_name)
            lbl.setFont(font)
            lbl.setAlignment(Qt.AlignCenter)
            labelLayout.addWidget(lbl)

        # Combine the slider and label layouts
        combinedLayout = QHBoxLayout()
        combinedLayout.addLayout(sliderLayout)
        combinedLayout.addLayout(labelLayout)

        return combinedLayout


    def create_3D_plane_buttonss(self):
        buttonXYZPlaneLayout = QVBoxLayout()
        # Create buttons
        self.buttonXY = QPushButton("XY")
        self.buttonYZ = QPushButton("YZ")
        self.buttonXZ = QPushButton("XZ")
        buttons = [self.buttonXY, self.buttonYZ, self.buttonXZ]
        for button in buttons:
            button.setCheckable(True)
            button.clicked.connect(self.on_button_clicked)
            button.setFixedSize(100, 40)  # Adjust the size as needed
            button.setStyleSheet("""
                QPushButton { font-size: 16pt; }
                QPushButton:checked { background-color: green; color: white; }
            """)
            buttonXYZPlaneLayout.addWidget(button)
        return buttonXYZPlaneLayout

    def on_button_clicked(self):
        sender = self.sender()
        if sender.isChecked():
            # Uncheck other buttons
            for button in [self.buttonXY, self.buttonYZ, self.buttonXZ]:
                if sender != button:
                    button.setChecked(False)

            # Determine which axis is selected and get the slider position
            if sender == self.buttonXY:
                position = self.slider3.value()
                # self.update_slice_plane('XY', position)
                self.update_volume_with_highlighted_slice('XY', position)
            elif sender == self.buttonYZ:
                position = self.slider1.value()
                # self.update_slice_plane('YZ', position)
                self.update_volume_with_highlighted_slice('YZ', position)
            elif sender == self.buttonXZ:
                position = self.slider2.value()
                # self.update_slice_plane('XZ', position)
                self.update_volume_with_highlighted_slice('XZ', position)



    def table_select(self, row, column):
        selected_id = self.idTable.item(row, 0).text()
        column_name = self.idTable.horizontalHeaderItem(column).text()
        print(f"Selected ID: {selected_id}, Column: {column_name}")

        try:
            selected_id=int(selected_id)
            self.tensor3D_current, self.header = utils.load_nii(file_type=column_name,
                                                                fileid=int(selected_id))
            self.tensor3D_current=self.tensor3D_current[:,:,:]
            self.updateImage(self.imageView1, self.slider1.value(), axis=0)
            self.updateImage(self.imageView2, self.slider2.value(), axis=1)
            self.updateImage(self.imageView3, self.slider3.value(), axis=2)
        except:
            self.tensor3D_current, self.header = utils.load_nii(file_type="OrigRes",
                                                                fileid=selected_id,
                                                                metabolic_str=self.metabolic_current)
            self.init_pos = np.array([self.tensor3D_current.shape[0]/2,
                                      self.tensor3D_current.shape[1]/2,self.
                                      tensor3D_current.shape[2]/2]).astype(int)
            self.updateFromSpinBox1(self.init_pos[0])
            self.updateFromSpinBox2(self.init_pos[1])
            self.updateFromSpinBox3(self.init_pos[2])

        # You can add additional logic here to do something with the selected ID and column
        self.update_3D_plot()

    def imageClicked(self, event):
        # Convert the mouse click position to image coordinates
        pos = self.imageView1.getImageItem().mapFromScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        # Handle the click event, for example, print coordinates
        print(f"Clicked coordinates: x={x}, y={y}")

        # Call the original mouse press event of the parent to maintain default interactive behaviors
        self.imageView1.getImageItem().parent().mousePressEvent(event)

    def updateLine23(self, value):
        # Update horizontal line position based on first slider
        self.vLine2.setPos(value)
        self.vLine3.setPos(value)

    def updateLine13(self, value):
        # Update vertical line position based on second slider
        self.vLine1.setPos(value)
        self.hLine3.setPos(value)

    def updateLine12(self, value):
        # Update vertical line position based on second slider
        self.hLine1.setPos(value)
        self.hLine2.setPos(value)

    def updateFromSpinBox1(self, value):
        self.spinBox1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.slider1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.spinBox1.setValue(value)
        self.slider1.setValue(value)
        self.updateLine23(value)
        self.updateImage(self.imageView1, value, axis=0)
        
    def updateFromSlider1(self, value):
        self.spinBox1.setValue(value)
        self.updateLine23(value)
        self.updateImage(self.imageView1, value, axis=0)

    def updateFromSpinBox2(self, value):
        self.spinBox2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.slider2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.spinBox2.setValue(value)
        self.slider2.setValue(value)
        self.updateLine13(value)
        self.updateImage(self.imageView2, value, axis=1)

    def updateFromSlider2(self, value):
        self.spinBox2.setValue(value)
        self.updateLine13(value)
        self.updateImage(self.imageView2, value, axis=1)


    def updateFromSpinBox3(self, value):
        self.spinBox3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.slider3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.spinBox3.setValue(value)
        self.slider3.setValue(value)
        self.updateLine12(value)
        self.updateImage(self.imageView3, value, axis=2)

    def updateFromSlider3(self, value):
        self.spinBox3.setValue(value)
        # self.slider3.setValue(value)
        self.updateLine12(value)
        self.updateImage(self.imageView3, value, axis=2)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select a NIfTI file", self.tensor3D_currentPATH,
                                                  "NIfTI Files (*.nii);;All Files (*)", options=options)
        if fileName:
            print("Selected file:", fileName)
            self.loadNiiFile(fileName)

    def on_metabolic_slider_value_changed(self, value):
        # Print the label corresponding to the current slider position
        if value >= 0 and value < len(self.metabolics_labels):
            print(self.metabolics_labels[value])

    def on_slider_value_changed(self, value):
        # Update the slice plane based on the currently checked button and slider value
        if self.buttonXY.isChecked():
            self.update_slice_plane('XY', self.slider3.value())
        elif self.buttonYZ.isChecked():
            self.update_slice_plane('YZ', self.slider1.value())
        elif self.buttonXZ.isChecked():
            self.update_slice_plane('XZ', self.slider2.value())

    def loadNiiFile(self, file_path):
        # Load the NIfTI file
        self.tensor3D_current = np.flip(nib.load(file_path).get_fdata(), axis=2)
        self.update_3D_plot()
        # Update the slider range
        self.slider.setRange(0, self.tensor3D_current.shape[self.current_axis] - 1)
        # Update the image
        self.updateImage(self.imageView1, 0, axis=0)
        self.updateImage(self.imageView2, 0, axis=1)
        self.updateImage(self.imageView3, 0, axis=2)

    def normalize_data(self,data):
        # Replace NaNs with 0 for safety
        data = data - np.nanmin(data)
        data = data / np.nanmax(data)
        data = np.nan_to_num(data)  # Convert NaNs to 0
        # Normalize data to the range 0-1
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val != 0:
            data = (data - min_val) / (max_val - min_val)
        else:
            # Avoid division by zero if all data values are the same
            data = np.zeros(data.shape)
        return data

    def create_vispy_canvas(self):
        tensor3D = self.normalize_data(self.tensor3D_current)
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = canvas.central_widget.add_view()

        # Create the volume visual
        self.volume = scene.visuals.Volume(tensor3D, parent=self.view.scene)

        # Create a slice plane
        self.slice_plane = scene.visuals.Plane(direction='+z', color=(0, 1, 0, 0.5), parent=self.view.scene)
        self.slice_plane.transform = STTransform(scale=(tensor3D.shape[1], tensor3D.shape[0], 1))
        self.slice_plane.visible = False

        self.view.camera = scene.cameras.TurntableCamera()
        return canvas.native

    def highlight_slice(self,tensor3D, axis, slice_index, highlight_color):
        modified_tensor = np.copy(tensor3D)

        # Assign the highlight color to the specified slice
        if axis == 'XY':
            modified_tensor[slice_index, :, :] = highlight_color
        elif axis == 'YZ':
            modified_tensor[:, slice_index, :] = highlight_color
        elif axis == 'XZ':
            modified_tensor[:, :, slice_index] = highlight_color

        return modified_tensor

    def update_volume_with_highlighted_slice(self, axis, position):
        # Normalize and highlight the slice
        normalized_data = self.normalize_data(self.tensor3D_current)
        highlighted_data = self.highlight_slice(normalized_data, axis, position, highlight_color=1)  # or any color value you choose

        # Update the volume data
        self.volume.set_data(highlighted_data)

    def update_slice_plane(self, axis, position):
        # Delete the existing slice plane
        if hasattr(self, 'slice_plane'):
            self.slice_plane.parent = None

        # Recreate the slice plane based on the selected axis
        if axis == 'XY':
            self.slice_plane = scene.visuals.Plane(direction='+z', color=(0, 1, 0, 0.5), parent=self.view.scene)
            self.slice_plane.transform = STTransform(translate=(0, 0, position))
        elif axis == 'YZ':
            self.slice_plane = scene.visuals.Plane(direction='+x', color=(0, 1, 0, 0.5), parent=self.view.scene)
            self.slice_plane.transform = STTransform(translate=(position, 0, 0))
        elif axis == 'XZ':
            self.slice_plane = scene.visuals.Plane(direction='+y', color=(0, 1, 0, 0.5), parent=self.view.scene)
            self.slice_plane.transform = STTransform(translate=(0, position, 0))

        self.slice_plane.visible = True

    def update_3D_plot(self):
        # Normalize the new data
        new_tensor3D = self.tensor3D_current
        normalized_data = self.normalize_data(new_tensor3D)
        normalized_data = normalized_data.astype(np.float32)
        self.volume.set_data(normalized_data)

    def updateImage(self, imageView, slice_index, axis):
        if axis == 0:
            slice_data = self.tensor3D_current[slice_index, :, :]
        elif axis == 1:
            slice_data = self.tensor3D_current[:, slice_index, :]
        elif axis == 2:
            slice_data = self.tensor3D_current[:, :, slice_index]

        # Create masks for different data conditions
        nan_mask = np.isnan(slice_data)
        minus_one_mask = (slice_data == -1)
        valid_data_mask = ~nan_mask & ~minus_one_mask

        # Prepare an empty array for normalized data
        normalized_data = np.zeros_like(slice_data)

        # Normalize valid data (ignoring NaNs and -1 values)
        if np.any(valid_data_mask):
            valid_min = np.nanmin(slice_data[valid_data_mask])
            valid_max = np.nanmax(slice_data[valid_data_mask])
            normalized_data[valid_data_mask] = 255 * (slice_data[valid_data_mask] - valid_min) / (valid_max - valid_min) + 1

        # Apply custom colormap
        colored_slice = self.custom_colormap[normalized_data.astype(np.int32)]

        # Setting voxels with value -1 to green (green in RGBA is [0, 255, 0, 255])
        colored_slice[minus_one_mask] = [0, 255, 0]

        # Update the image view
        imageView.setImage(colored_slice)



def main():
    app = QApplication(sys.argv)
    # plot3DWindow = Plot3DWindow()
    # plot3DWindow.show()
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()




