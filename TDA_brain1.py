"""
Trial num 1 of the use of TDA for brain micro-ct data analysis
Using pyvista, gitto, gudhi, and the classics
"""
############################################
#import packages
import pyvista as pv
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import velour


from gtda.homology import WeakAlphaPersistence, VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance



from sklearn.metrics import pairwise_distances

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot


# gtda plotting functions
from gtda.plotting import plot_heatmap

from gtda.pipeline import Pipeline

############################################





#Extract the mesh from the ply file
mesh = pv.read('Images/Mesh.ply')
X = mesh.points
#Sampling from data


N = 1000
X=X[np.random.choice(len(X), size=N, replace=False)]


# gtda pipeline

# Data wrangling
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

# Define filter function – can be any scikit-learn transformer
filter_func = Projection(columns=[0, 1])
# Define cover
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
# Choose clustering algorithm – default is DBSCAN
clusterer = DBSCAN()

# Configure parallelism of clustering step
n_jobs = 1

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs,
)





 




fig = plot_static_mapper_graph(pipe, X, layout_dim=3)
fig.show(config={'scrollZoom': True})

print("Initialization done")

from gtda.homology import VietorisRipsPersistence







print("Computing VietorisRips persistence")


VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Connected components, holes, void
#WA = WeakAlphaPersistence(homology_dimensions=(0, 1, 2))
X_reshaped= X.reshape(1, *X.shape)
diagrams = VR.fit_transform(X_reshaped)
diagrams.shape

from gtda.plotting import plot_diagram

"""i = 0

plot = plot_diagram(diagrams[i])

plot.show()"""













from gtda.diagrams import PersistenceEntropy

'''PE = PersistenceEntropy()
features = PE.fit_transform(diagrams)

# expect shape - (n_point_clouds, n_homology_dims)
print("n_point_clouds, n_homology_dims",features)


fig = plot_point_cloud(features)
fig.show()
'''










'Rescaling diagrams'

window_number =  0

diagramScaler = Scaler()

X_scaled = diagramScaler.fit_transform(diagrams)

print("scaled diagram")
plot = diagramScaler.plot(X_scaled, sample=window_number)

plot.show()











'Betti curve'



BC = BettiCurve()

X_betti_curves = BC.fit_transform(X_scaled)

fig = BC.plot(X_betti_curves, sample=window_number)
fig.show()











PE = PersistenceEntropy()

X_persistence_entropy = PE.fit_transform(X_scaled)

print("n_point_clouds, n_homology_dims",X_persistence_entropy)


fig = plot_point_cloud(X_persistence_entropy)
fig.show()

"""fig = px.line(title='Persistence entropies, indexed by sliding window number')
for dim in range(X_persistence_entropy.shape[1]):
    fig.add_scatter(y=X_persistence_entropy[:, dim], name=f"PE in homology dimension {dim}")
fig.show()
"""




# Try different distances
#print("Trying different distances")
#Landscape L2 distance
#p_L = 2
#n_layers = 5
#PD = PairwiseDistance(metric='landscape',
#                      metric_params={'p': p_L, 'n_layers': n_layers, 'n_bins': 1000},
#                      order=None)

#X_distance_L = PD.fit_transform(X_scaled)
#X_distance_L.shape


#fig = plot_heatmap(X_distance_L[:, :, 0], colorscale='reds')
#fig.show()



# 2-Wasserstein distances

#p_W = 2
#PD = PairwiseDistance(metric='wasserstein',
#                      metric_params={'p': p_W, 'delta': 0.1},
#                      order=None)

#X_distance_W = PD.fit_transform(X_scaled)

#fig = plot_heatmap(X_distance_W[:, :, 0], colorscale='greens')
#fig.show()

# Geodesic distance

#n_neighbors = 2
#kNN = KNeighborsGraph(n_neighbors=n_neighbors)

#X_kNN = kNN.fit_transform(X_scaled)

#GGD = GraphGeodesicDistance()

#GGD.fit_transform_plot(X_kNN, sample=window_number);

#fig = plot_heatmap(pairwise_distances(X_scaled[window_number]), colorscale='blues')
#fig.show()




#Try persistent diagram with new distance
# Steps of the Pipeline
#steps = [
#    ('kNN_graph', kNN),
#    ('graph_geo_distance', GGD),
#    ('diagrams', VietorisRipsPersistence(metric='precomputed',
#                                         homology_dimensions=(0,1,2)))
#    ]

# Define the Pipeline
#pipeline = Pipeline(steps)

# Run the pipeline
#X_diagrams = pipeline.fit_transform(X_scaled)

#fig = pipeline[-1].plot(X_diagrams, sample=window_number)

#fig.show()