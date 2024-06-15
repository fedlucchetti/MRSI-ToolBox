import os, sys, json
import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from tools.debug import Debug
import subprocess

debug = Debug()

class TractTools:
    def __init__(self):
        pass

    def tck2trk(self,anatomy, tractogram, force=False):
        try:
            nii = nib.load(anatomy)
        except Exception as e:
            raise ValueError("Expecting anatomy image as first argument. Error: {}".format(e))
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TckFile:
            debug.error("Skipping non TCK file: '{}'".format(tractogram))
            return
        output_filename = tractogram.replace(".tck",".trk")
        if os.path.isfile(output_filename) and not force:
            debug.error("Skipping existing file: '{}'. Use force=True to overwrite.".format(output_filename))
            return

        header = {}
        header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
        header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
        header[Field.DIMENSIONS] = nii.shape[:3]
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))
        tck = nib.streamlines.load(tractogram)
        nib.streamlines.save(tck.tractogram, output_filename, header=header)
        debug.success(f"Converted to {output_filename}")



    def take_screenshot(self,tck_file, screenshot_file, anatomy_file=None):
        # Prepare the base mrview command
        cmd = ['mrview', tck_file, '-capture.grab', screenshot_file]

        # Add optional anatomy overlay if provided
        if anatomy_file:
            cmd.extend(['-overlay.load', anatomy_file])
        
        # Execute the command
        subprocess.run(cmd, check=True)
        print(f"Screenshot saved to {screenshot_file}")

    def exit_run(self,progress_path,data,exit=True):
        with open(progress_path, 'w') as file:
            json.dump(data, file, indent=4)
        if exit:
            sys.exit()


if __name__=="__main__":
    filename = "smallerTracks_200k.tck"
    dirpath  = "/media/veracrypt2/Connectome/Data/Mindfulness-Project/derivatives/tractography/sub-S001/ses-V1"
    filepath = os.path.join(dirpath,filename)
    outfilepath = filepath.replace(".tck",".png")
    # tcktools = TractTools()
    # tcktools.take_screenshot(filepath,filepath.replace(".trk",".png"))



    import nibabel as nib
    from dipy.io.streamline import load_tck
    from fury import window, actor
    import nibabel as nib
    import numpy as np
    # from dipy.tracking.streamline import remove_invalid_streamlines
    # Create a simple NIfTI image as a reference (e.g., a 3D array of zeros)
    data          = np.ones((128, 128, 128))
    affine        = np.eye(4)
    reference_img = nib.Nifti1Image(data, affine)

    # Save the reference image to a file
    reference_file = 'reference_image.nii.gz'
    nib.save(reference_img, reference_file)

    # Load the .tck file
    tck_file = filepath
    # streamlines = load_tck(tck_file).streamlines
    # streamlines = load_tck(tck_file, reference=reference_img)
    tractogram = load_tck(tck_file, reference=reference_img, bbox_valid_check=False)

    # Remove invalid streamlines
    # tractogram = remove_invalid_streamlines(tractogram)
    streamlines = tractogram.streamlines
    # Create a scene
    scene = window.Scene()

    # Create an actor for the streamlines
    streamlines_actor = actor.line(streamlines)

    # Add the actor to the scene
    scene.add(streamlines_actor)

    # Show the scene in an interactive window
    window.show(scene)

    # Save a screenshot of the scene
    screenshot = window.snapshot(scene, fname=outfilepath, size=(800, 800))

    print("Screenshot saved to tractography_screenshot.png")


# Example usage:
# converter = TractogramConverter(anatomy='path_to_anatomy.nii.gz', tractograms=['path_to_tractogram1.tck', 'path_to_tractogram2.tck'], force=True)
# converter.tck2trk()
