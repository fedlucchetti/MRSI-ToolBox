import matplotlib 
# matplotlib.use('Agg')  # Switch to the Agg backend
import matplotlib.pyplot as plt
import numpy as np
from tools.debug import Debug

from os.path import join ,split
debug  = Debug()
class PlotSlices:
    def __init__(self) -> None:
        pass

    def find_boundaries(self,mask, axis=0):
        """
        Find the minimum and maximum boundaries along a specified axis of a 3D mask.

        Parameters:
        - mask: np.ndarray, a 3D numpy array representing the mask.
        - axis: int, the axis along which to find the boundaries (0, 1, or 2).

        Returns:
        - A tuple containing the minimum and maximum indices along the specified axis
        where the mask transitions from 0 to 1. Returns (None, None) if no 1s are found.
        """
        assert mask.ndim == 3, "The mask must be a 3D numpy array."
        assert axis in [0, 1, 2], "Axis must be 0, 1, or 2."
        
        # Check if there's at least one 1 in the mask
        if np.any(mask == 1):
            # Find indices where mask is 1
            ones_indices = np.argwhere(mask == 1)
            
            # Extract the indices along the specified axis
            specific_axis_indices = ones_indices[:, axis]
            
            # Find min and max along the specified axis
            min_boundary = specific_axis_indices.min()
            max_boundary = specific_axis_indices.max()
            
            return min_boundary, max_boundary
        else:
            return None, None

    def plot_img_slices(self,images, slice_percentages, titles=None, outpath=None, PLOTSHOW=False, mask=None):
        """
        Plots slices from multiple 3D images at given percentages of the Z dimension within computed boundaries,
        with titles on the left.

        Parameters:
        - images: List of 3D numpy arrays representing the images.
        - slice_percentages: List of percentages (0-100) of the Z dimension to plot within boundaries.
        - titles: List of title prefixes for each image. If None, defaults to 'ImgN Slice'.
        - outpath: Path to save the output figure. If None, figure is not saved.
        - PLOTSHOW: Boolean, if True, shows the plot.
        - z_boundaries: Tuple of (min_z, max_z) boundaries within which to rescale the percentages. If None, use full range.
        """
        if mask is not None:
            z_boundaries = self.find_boundaries(mask,axis=2)
        else: z_boundaries=None
        titles = titles or [f'Img{idx+1} Slice' for idx in range(len(images))]
        titles = titles[::-1]
        num_images = len(images)
        num_slices = len(slice_percentages)
        fig, axes = plt.subplots(num_images, num_slices, figsize=(num_slices * 3, num_images * 3))

        # Ensure axes is a 2D array for easy indexing if there's only one image or one slice
        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_slices == 1:
            axes = np.expand_dims(axes, axis=-1)

        for img_idx, img in enumerate(images):
            # Adjust slice indices based on z_boundaries
            min_z, max_z = z_boundaries if z_boundaries else (0, img.shape[-1] - 1)
            z_slices = [int(min_z + (max_z - min_z) * (p / 100.0)) for p in slice_percentages]

            for slice_idx, z in enumerate(z_slices):
                axes[img_idx, slice_idx].imshow(img[:, :, z])
                axes[img_idx, slice_idx].axis('off')  # Hide axes ticks

                # Set title for the row
                title_prefix = titles[img_idx] if titles else f'Img{img_idx + 1} Slice'
                # Use fig.text() to place titles on the left-hand side vertically
                fig.text(0.01, 0.5 * (2 * img_idx + 1) / num_images, title_prefix, va='center', rotation='vertical', fontweight='bold', fontsize=16, color='red')

        plt.tight_layout()
        if PLOTSHOW:
            plt.show()
        if outpath is not None:
            fig.savefig(f'{outpath}.pdf')
            plt.close(fig)
        else:
            plt.close(fig)


 