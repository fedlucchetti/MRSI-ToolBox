import torch
import numpy as np
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    ToTensor,
    Resize,
    ScaleIntensity,
)
from monai.inferers import sliding_window_inference
from monai.transforms import Resize
from vispy import app, scene

from Dev.Analytics.tools.datautils import DataUtils
from tools.debug import Debug

debug = Debug()
utils = DataUtils()



class Segment(object):
    def __init__(self) -> None:
        self.model = UNETR(
            in_channels=1,
            out_channels=2,  # Assuming binary classification
            img_size=(128, 128, 64),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
        ).to(torch.device('cpu'))
        self.model.eval()

    def reshape(self,tensor3D,outshape = (128, 128, 64)):
        tensor3D = tensor3D.unsqueeze(0)  # Shape becomes [1, D, H, W]
        resize_transform = Resize(spatial_size=outshape)
        return resize_transform(tensor3D)

    def forward(self,tensor3D,mask):
        tensor3D[np.isnan(tensor3D)] = 0
        tensor3D=tensor3D/tensor3D.max()
        original_shape = tensor3D.shape
        # If tensor3D is a numpy array
        if isinstance(tensor3D, np.ndarray):
            tensor3D = torch.from_numpy(tensor3D)
        if isinstance(mask, np.ndarray):
            mask     = torch.from_numpy(mask)

        tensor3D *= mask
        debug.info("tensor3D shape",tensor3D.shape)
        tensor3D = self.reshape(tensor3D, outshape = (128, 128, 64))
        mask     = self.reshape(mask,     outshape = (128, 128, 64))

        with torch.no_grad():
            outputs = sliding_window_inference(tensor3D.unsqueeze(0), (128, 128, 64), 4, self.model)
            result = torch.argmax(outputs, dim=1).squeeze(0)
            outputs = outputs.squeeze(0)

        result_np  = result.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()
        mask_np    = mask.detach().cpu().numpy()
        debug.info("forward result_np shape",result_np.shape)  
        debug.info("forward outputs_np shape",outputs_np.shape)  
        debug.info("forward mask_np shape",mask_np.shape)  

        for idt, tensor3D in enumerate(outputs_np):
            outputs_np[idt] = tensor3D * mask_np[0]
        return outputs_np, result_np[0]*mask_np[0]
    



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


if __name__=="__main__":
    segment = Segment()
    tensor3D, _ = utils.load_nii("Basic", 1)
    mask, _ = utils.load_nii("Qmask", 1)
    outputs_np, result_np = segment.forward(tensor3D,mask)
    debug.info("outputs_np.shape",outputs_np.shape)
    debug.info("result_np.shape",result_np.shape)
    # tensor3D = segment.normalize_data(result_np)
    # Create a canvas with a 3D viewport
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # Set up volume visual using the custom colormap
    volume = scene.visuals.Volume(result_np, parent=view.scene)
    # Set the camera to the turntable camera
    view.camera = scene.cameras.TurntableCamera()
    app.run()





