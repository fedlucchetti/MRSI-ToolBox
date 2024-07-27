import matplotlib 
# matplotlib.use('Agg')  # Switch to the Agg backend

import matplotlib.pyplot as plt
import numpy as np
from os.path import join ,split
from matplotlib.colors import LinearSegmentedColormap
from graphplot.colorbar import ColorBar

colorbar = ColorBar()

class SimMatrixPlot:
    def __init__(self) -> None:
        pass


    def plot_simmatrix(self, correlation_matrix, ax=None, parcel_ids_positions=None, 
                       colorbar_label="Correlation Strength",parcel_labels=None, show_parcels="VH", titles=None, result_path=None, 
                       colormap="plasma",scale_factor=1, dpi=100, show_colorbar=True):
        """
        Plots a similarity matrix on the provided axes. If no axes are provided, creates a new figure and axes.

        Parameters:
        - ax: Matplotlib axes object to plot on. If None, creates a new figure.
        - parcel_ids_positions: Dictionary with parcel ids as keys and tuple (min_val, max_val) as values.
        - show_parcels: String "H" to show horizontal labels, "V" for vertical, "VH" for both.
        - titles: Title of the plot.
        - result_path: Path to save the plot.
        - colormap: Color map for the plot.
        - scale_factor: Factor to scale the plot size.
        - dpi: Dots per inch for the plot resolution.
        - show_colorbar: Boolean to show the colorbar.
        """
        colormap=colorbar.bars(colormap)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12 * scale_factor, 10 * scale_factor), dpi=dpi)
        else:
            fig = ax.figure

        cax = ax.matshow(correlation_matrix, interpolation='nearest', cmap=colormap)
        ax.grid(False)

        if parcel_ids_positions:
            # Create lists for positions and labels
            middle_positions = [(min_val + max_val) / 2 for min_val, max_val in parcel_ids_positions.values()]
            min_positions = [min_val for min_val, _ in parcel_ids_positions.values()]
            max_positions = [max_val for _, max_val in parcel_ids_positions.values()]
            labels = list(parcel_ids_positions.keys())

            # Combine and sort positions and labels for ticks
            combined_positions = min_positions + middle_positions + max_positions
            combined_labels = [""] * len(min_positions) + labels + [""] * len(max_positions)

            # Sort positions and labels
            sorted_indices = np.argsort(combined_positions)
            sorted_positions = np.array(combined_positions)[sorted_indices]
            sorted_labels = np.array(combined_labels)[sorted_indices]
            
            if show_parcels in ["H", "VH"]:
                ax.set_xticks(sorted_positions)
                ax.set_xticklabels(sorted_labels, rotation=45, fontsize=14*scale_factor, fontweight='bold')

            if show_parcels in ["V", "VH"]:
                ax.set_yticks(sorted_positions)
                ax.set_yticklabels(sorted_labels, fontsize=14*scale_factor, fontweight='bold')
        elif parcel_labels is not None:
            positions = np.arange(0,len(parcel_labels))
            if show_parcels in ["H", "VH"]:
                ax.set_xticks(positions)
                ax.set_xticklabels(parcel_labels, rotation=45, fontsize=14*scale_factor, fontweight='bold') 
            if show_parcels in ["V", "VH"]:         
                ax.set_yticks(positions)
                ax.set_yticklabels(parcel_labels, fontsize=14*scale_factor, fontweight='bold')    

        if titles:
            ax.set_title(titles, fontsize=18, fontweight='bold')
        if show_colorbar:
            fig.colorbar(cax, ax=ax, fraction=0.1 * scale_factor, pad=0.04, label=colorbar_label)

        if result_path:
            fig.savefig(f"{result_path}.pdf", dpi=dpi)

        return fig, ax  # Always return fig and ax for further manipulation

    def plot_multiple_simmatrices(self,simmatrices, titles=None,result_path=None,plotshow=False, cmap='plasma',dpi=100):
        """
        Plots multiple similarity matrices on the same figure.

        Parameters:
            simmatrices (list of np.ndarray): List of similarity matrices to plot.
            titles (list of str, optional): List of titles for each subplot.
            cmap (str, optional): Colormap to use for the plots.
        """
        num_matrices = len(simmatrices)
        # Ensure titles are provided or create default titles
        if titles is None:
            titles = [f'Matrix {i + 1}' for i in range(num_matrices)]
        if len(titles) != num_matrices:
            raise ValueError("The number of titles must match the number of matrices.")
        # Determine grid size (e.g., 2x2 for 4 matrices)
        grid_size = int(np.ceil(np.sqrt(num_matrices)))
        # Create subplots
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        # Flatten the array of axes to easily iterate over it
        axs = axs.flatten()
        for idx, (matrix, title) in enumerate(zip(simmatrices, titles)):
            im = axs[idx].imshow(simmatrices[idx], cmap=cmap)
            axs[idx].set_title(title)
            axs[idx].axis('off')
            fig.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)
        # Remove extra subplots if any
        for extra_ax in axs[num_matrices:]:
            extra_ax.axis('off')
        plt.tight_layout()
        if result_path:
            fig.savefig(f"{result_path}.pdf", dpi=dpi)  # Ensure dpi is set for saving as well
        if plotshow:
            plt.show()
        plt.close(fig)


    def __plot_simmmatrix(self, correlation_matrix,parcel_labels, titles=None, outpath=None, PLOTSHOW=False):
        """
        Plots slices from multiple 3D images at given percentages of the Z dimension, with titles on the left.

        Parameters:
        - images: List of 3D numpy arrays representing the images.
        - slice_percentages: List of percentages (0-100) of the Z dimension to plot.
        - titles: List of title prefixes for each image. If None, defaults to 'ImgN Slice'.
        """
        fig, axs = plt.subplots(figsize=(20, 4))  # Adjust figsize as needed

        # Plot each 2D correlation matrix on its own subplot
        im = axs.imshow(correlation_matrix[:, :], aspect='auto')
        
        # Optional: Add a title to each subplot indicating the metabolite
        axs.title.set_text(f'{titles}')
        axs.set_xticklabels(parcel_labels, rotation=90)  # Rotate x labels for better readability
        axs.set_yticklabels(parcel_labels)
        # Optional: Add a colorbar to each subplot to indicate the scale
        fig.colorbar(im, ax=axs)

        plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
        if PLOTSHOW:
            plt.show()
        if outpath is not None:
            fig.savefig(f'{outpath}.pdf')
            plt.close(fig)
        else:
            plt.close(fig)

 