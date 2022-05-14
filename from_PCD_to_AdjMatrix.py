import sys
from optparse import OptionParser, OptionGroup
import numpy as np 
import igraph
import tifffile as TIF

import gtda
from gtda.plotting import plot_heatmap
from gtda.images import Binarizer


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


from gtda.mapper.filter import Projection
from gtda.mapper.cover import CubicalCover
# scikit-learn method
from sklearn.cluster import DBSCAN
# giotto-tda method
from gtda.mapper.cluster import FirstSimpleGap



parser = OptionParser()
parser.add_option("--input", dest='input', help='Location of the .tiff file containing the image you need to analyse')
(options, args) = parser.parse_args()

# Import the tiff file with tifffile

#print(options.input)
img = TIF.imread(options.input)

#Add new dimension to the obtained np ndarray to shape the data as giotto likes it

img = img[np.newaxis]


# Binarize the image 

binarizer = Binarizer(threshold=0.8)
im_binarized = binarizer.fit_transform(img)

# From binarized image to PCD
PCD = gtda.images.ImageToPointCloud()
#PCD.fit_transform_plot(im_binarized)
pcd_of_the_image = PCD.fit_transform(im_binarized)




#Use giotto to use the MApper algorithm with DBSCAN on the pcd 

# Define filter function – can be any scikit-learn transformer
filter_func = Projection(columns=[0, 1])
# Define cover
cover = CubicalCover(n_intervals=10, overlap_frac=0.1
                    )
# Choose clustering algorithm – default is DBSCAN
clusterer = DBSCAN()

# Configure parallelism of clustering step
n_jobs = -1 # -1 to use all the cores

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs,
)

# create the graph

graph = pipe.fit_transform(pcd_of_the_image[0])


graph.write_adjacency(options.input +'_adj_matrix')
