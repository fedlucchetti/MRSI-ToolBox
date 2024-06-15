# Copyright (C) 2009-2022, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

"""This module defines the `showmatrix_gpickle` script that loads and displays a connectivity matrix."""

import sys
import os
from itertools import cycle

import networkx as nx
import numpy as np
import copy

import matplotlib.colors as colors

# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib.pyplot import imshow, cm, show, figure, colorbar, hist
import matplotlib.pyplot as plt
import matplotlib.path as m_path
import matplotlib.patches as m_patches
from scipy.stats import pareto

class SpringNetwork:
    def __init__(self):    
        pass

    def plot_weighted_graph(self,simmatrix, labels):
        """
        Plots a graph based on a similarity matrix and node labels using the
        Fruchterman-Reingold layout.

        Parameters:
        - simmatrix: A 2D numpy array where each element represents the similarity (weight) between nodes.
        - labels: A list of labels for each node.
        """
        G = nx.Graph()

        # Add nodes with labels directly as identifiers
        for label in labels:
            G.add_node(label)

        # Add edges with weights from the similarity matrix
        for i in range(len(simmatrix)):
            for j in range(i+1, len(simmatrix)):
                # Add an edge only if there's a non-zero similarity
                if simmatrix[i][j] > 0:
                    G.add_edge(labels[i], labels[j], weight=simmatrix[i][j])
        # Apply the Fruchterman-Reingold layout with weight consideration
        pos = nx.spring_layout(G, weight='weight')

        # Draw the graph
        plt.figure(figsize=(10, 10))  # Set the size of the graph
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', 
                width=1, font_size=10, node_size=500, alpha=0.9)

        plt.title("Weighted Network Graph using Fruchterman-Reingold Layout")
        plt.axis('off')  # Turn off the axis
        plt.show()

    
if __name__=="__main__":
    size=8
    pltspring = SpringNetwork()
    simmatrix = pltspring.generate_pareto_similarity_matrix(size=size, shape=2.62)
    labels    = np.arange(0,size).astype(str)
    pltspring.plot_weighted_graph(simmatrix, labels)
