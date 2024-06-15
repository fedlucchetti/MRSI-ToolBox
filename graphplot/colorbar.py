from matplotlib.colors import LinearSegmentedColormap


class ColorBar():
    def __init__(self):
        pass

    def bars(self,color_bar="blueblackred"):
        if color_bar=="blueblackred":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redblackblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="bluewhitered":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redwhiteblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="bbo":
            colors = [
                (0, "dodgerblue"),  # Blue for lowest values
                (0.25, "darkblue"),
                (0.5, "black"),  # Black for middle values
                (0.75, "darkred"),  # Black for middle values
                (1, "orangered")  # Red for highest values
            ]
        elif color_bar=="obb":
            colors = [
                (0, "orangered"),  # Blue for lowest values
                (0.25, "darkred"),
                (0.5, "black"),  # Black for middle values
                (0.75, "darkblue"),  # Black for middle values
                (1, "dodgerblue")  # Red for highest values
            ]
        elif color_bar=="bwo":
            colors = [
                (0, "darkblue"),  # Blue for lowest values
                (0.25, "dodgerblue"),
                (0.5, "white"),  # Black for middle values
                (0.75, "orangered"),  # Black for middle values
                (1, "darkred")  # Red for highest values
            ]
        elif color_bar=="owb":
            colors = [
                (0, "darkred"),  # Blue for lowest values
                (0.25, "orangered"),
                (0.5, "white"),  # Black for middle values
                (0.75, "dodgerblue"),  # Black for middle values
                (1, "darkblue")  # Red for highest values
            ]
        else: 
            return color_bar
        return LinearSegmentedColormap.from_list("custom_cmap", colors)