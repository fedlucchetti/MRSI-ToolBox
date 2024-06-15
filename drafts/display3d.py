
import numpy as np
from vispy import app, scene
from Dev.Analytics.tools.datautils import DataUtils
from tools.debug import Debug
from vispy.color import Colormap

# from tools.ClickableIMG import ClickableImageView 

utils = DataUtils()
debug = Debug()

def create_colormap(data):
    # Create a colormap that transitions from transparent to opaque
    # Modify these points to adjust the appearance of your volume rendering
    data_mean = data.mean()
    data_std  = data.std()

    low_opacity_point = data_mean
    high_opacity_point = data_mean + data_std  # or choose a different point as needed

    # Define the colormap
    colors = [(0, 0, 0, 0),  # Full transparency at the minimum
            (low_opacity_point, low_opacity_point, low_opacity_point, 0.5),  # Mid-level opacity
            (high_opacity_point, high_opacity_point, high_opacity_point, 1)]  # Full opacity

    return Colormap(colors)

def normalize_data(data):
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

tensor3D, header = utils.load_nii(file_type="Basic",id=1)
tensor3D = normalize_data(tensor3D)
# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

cmap = create_colormap(tensor3D)

# Set up volume visual using the custom colormap
volume = scene.visuals.Volume(tensor3D, parent=view.scene)

# Set the camera to the turntable camera
view.camera = scene.cameras.TurntableCamera()

# Run the application
if __name__ == '__main__':
    app.run()
