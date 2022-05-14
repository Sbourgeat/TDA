from ssl import Options
import sys
from optparse import OptionParser, OptionGroup

from click import option
import numpy as np 
import igraph
import tifffile as TIF

import gtda
from gtda.plotting import plot_heatmap
from gtda.images import Binarizer
import matplotlib.pyplot as plt

# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module

import pandas as pd
import seaborn as sns
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

from gtda.images import RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler
from gtda.diagrams import HeatKernel



parser = OptionParser()
parser.add_option("--input", dest='input', help='Location of the .tiff file containing the image you need to analyse')
(options, args) = parser.parse_args()

from PIL import Image

# Import the tiff file with tifffile

#print(options.input)
print("Importing image" + options.input)

img = TIF.imread(options.input)
#img = Image.open(options.input)

#img = img[0]   
plt.imshow(img)

X = img.reshape(1, *img.shape)


import numpy as np
from gtda.plotting import plot_heatmap
from gtda.images import Binarizer

print('Binarization')
binarizer = Binarizer(threshold=0.1)
im_binarized = binarizer.fit_transform(1 - X)
plot_heatmap(im_binarized[0])
plt.show()



import gtda
print("Generating point cloud data")
PCD = gtda.images.ImageToPointCloud()
pced = PCD.fit_transform(im_binarized)
X = pced[0]

N = 1000
X=X[np.random.choice(len(X), size=N, replace=False)]
#gtda.plotting.plot_point_cloud(pced)

gtda.plotting.plot_point_cloud(pced[0])


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




from gtda.homology import WeakAlphaPersistence, VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance

homology_dimensions = (0, 1, 2)






print("Filtering the data")



# Define filter function – can be any scikit-learn transformer
filter_func = Projection(columns=[0, 1])
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






print('Using Mapper algo')




fig = plot_static_mapper_graph(pipe, X, layout_dim=3)
fig.show(config={'scrollZoom': True})



from gtda.homology import VietorisRipsPersistence

print('Computing VR complex persistence diagram')

VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
diagrams = VR.fit_transform(X.reshape(1, *X.shape))
diagrams.shape




from gtda.plotting import plot_diagram 

i=0
plot= plot_diagram(diagrams[i])

plot.show()





from gtda.diagrams import PersistenceEntropy



persistence_entropy = PersistenceEntropy()

# calculate topological feature matrix
X_basic = persistence_entropy.fit_transform(diagrams)

# expect shape - (n_point_clouds, n_homology_dims)
X_basic.shape

print("The entropy is", X_basic)
#np.save('Results/' + options.input + "res.txt", X_basic)




window_number =  0

diagramScaler = Scaler()

X_scaled = diagramScaler.fit_transform(diagrams)

print("scaled diagram")
plot = diagramScaler.plot(X_scaled, sample=window_number)

plot.show()



BC = BettiCurve()

X_betti_curves = BC.fit_transform(X_scaled)

fig = BC.plot(X_betti_curves, sample=window_number)
fig.show()


















