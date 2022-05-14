from ssl import Options
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

# Import the tiff file with tifffile

#print(options.input)
img = TIF.imread(options.input)


from sklearn.pipeline import Pipeline
from gtda.diagrams import Amplitude
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import PersistenceEntropy
from gtda.images import HeightFiltration


height_pipeline = Pipeline([
    ('binarizer', Binarizer(threshold=0.6)),
    ('filtration', HeightFiltration()),
    ('diagram', CubicalPersistence()),
    ('feature', PersistenceEntropy(nan_fill_value=-1)),
    
])


im_pipeline = height_pipeline.fit_transform(img)

df = pd.DataFrame(im_pipeline)
df.to_csv('Results/802.csv', index=True)

sns.distplot(df[0], label='H0')
sns.distplot(df[1], label='H1')
plt.xlabel('Persistence entropy')
plt.legend()
plt.show()


