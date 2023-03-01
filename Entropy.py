

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


import pyvista as pv
import ipywidgets 
import igraph
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import velour

from sklearn.pipeline import make_pipeline



class Entropy:
   



    def __init__(self, data, it) -> None:
        self.data = data
        self.it = it





    def mean_entropy(data, it) -> list :
        steps = [VietorisRipsPersistence(homology_dimensions=[0, 1, 2]),
                    PersistenceEntropy()]
        pipeline = make_pipeline(*steps)

        Mean_entropy=[]

        for i in range(it):
            Entropy = pipeline.fit_transform(data)
            Mean_entropy.append(Entropy)

        return np.average(Mean_entropy, axis=0)


    if __name__ == "__main__":
        mean_entropy(self.data,self.it)