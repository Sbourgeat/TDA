import sys
from optparse import OptionParser, OptionGroup
import numpy as np 
import igraph
import tifffile as TIF

import gtda
from gtda.plotting import plot_heatmap
from gtda.images import Binarizer

import pyvista as pv
import ipywidgets 
import igraph
import matplotlib.pyplot as plt

# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


from gtda.mapper.filter import Projection, Eccentricity, Entropy

# scikit-learn method
from sklearn.cluster import DBSCAN
# giotto-tda method
from gtda.mapper.cluster import FirstSimpleGap



parser = OptionParser()
parser.add_option("--input", dest='input', help='Location of the .ply file containing the mesh you need to analyse')
(options, args) = parser.parse_args()

# Import the tiff file with tifffile

#print(options.input)
mesh = pv.read(options.input)
cpos = mesh.plot()

X = mesh.points
print(len(X))
N = 1000
X=X[np.random.choice(len(X), size=N, replace=False)]
X = X.reshape(1, *X.shape)



# Define filter function – can be any scikit-learn transformer
filter_func = Projection([0,2])
# Define cover
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
# Choose clustering algorithm – default is DBSCAN
clusterer = DBSCAN()

# Configure parallelism of clustering step
n_jobs = -1

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs,
)


fig = plot_static_mapper_graph(pipe, X[0], layout_dim=3)
fig.show(config={'scrollZoom': True})



graph = pipe.fit_transform(X[0])

W = graph.get_edge_dataframe()

XX = graph.get_adjacency()

XX = np.array(XX.data)


L = list(dict.fromkeys(W['source'].tolist()))


for i in L:
    #print(i)
    w = W[W['source']==i]
    for ii in (w['target'].tolist()):
        #print(w[w['target']==ii]['weight'].tolist()[0])
        XX[i,ii] = w[w['target']==ii]['weight'].tolist()[0]
        #print(' ')
        


XX = np.where(XX ==0, np.inf, XX)

XX =  XX.reshape(1, *XX.shape)



import numpy as np
from numpy.random import default_rng
rng = default_rng(42)  # Create a random number generator

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence

from igraph import Graph

from IPython.display import SVG, display





# Instantiate topological transformer
VR = VietorisRipsPersistence(metric='precomputed')

# Compute persistence diagrams corresponding to each graph in X
diagrams = VR.fit_transform(XX)



plot = VR.plot(diagrams)
plot.show()
