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
from gtda.plotting import plot_point_cloud


import pyvista as pv
import ipywidgets 
import igraph
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import velour


import plotly.express as px
import pandas as pd
from sklearn.cluster import AffinityPropagation


PE = np.array([[9.82027879, 7.46868098, 1.2074247 ],
       [9.82275687, 7.38500095, 1.47979616],
       [9.8140593 , 7.33074124, 1.25749965],
       [9.81906141, 7.40960017, 1.63268977],
       [9.81576401, 7.50494039, 1.88439233],
       [9.81261865, 7.46265608, 1.85150513],
       [9.83508141, 7.34770496, 1.80602032],
       [9.8245328 , 7.56247448, 1.32849086],
       [9.81448857, 7.47247961, 1.52253721],
       [9.79925092, 7.31689832, 2.06688141],
       [9.80486463, 7.41738688, 2.64026878],
       [9.82180028, 7.39751705, 1.42904619],
       [9.81918856, 7.32592312, 1.58579437],
       [9.82386711, 7.38322296, 2.44156809],
       [9.82648781, 7.45809361, 1.96676235],
       [9.82342342, 7.38969619, 1.40184243],
       [9.80014728, 7.5705397 , 1.91827951],
       [9.81099179, 7.43702728, 2.37414987],
       [9.82880363, 7.47571608, 1.10212667],
       [9.78761771, 7.36716618, 1.08096558],
       [9.80955846, 7.45127795, 1.78389491],
       [9.81275515, 7.41149274, 1.43282934],
       [9.82340562, 7.31720958, 1.71617734],
       [9.8135433 , 7.40502995, 1.64600334],
       [9.83198161, 7.37043323, 1.76721218],
       [9.81719314, 7.3664142 , 1.35203565],
       [9.80491504, 7.33402976, 2.37396929],
       [9.80829648, 7.28676911, 2.31748165],
       [9.80635078, 7.39827204, 1.92831262],
       [9.81948937, 7.43166083, 1.48467639],
       [9.81642072, 7.48427329, 1.89911344],
       [9.81701039, 7.48509374, 1.50772574],
       [9.80941179, 7.47229697, 1.73163364],
       [9.80960915, 7.39754819, 2.02248911],
       [9.81604641, 7.51473486, 1.28609867],
       [9.81412409, 7.37954804, 1.33629429],
       [9.78902127, 7.36374341, 1.5741909 ],
       [9.79386298, 7.50543753, 1.70992583],
       [9.81350883, 7.45430142, 2.43433181],
       [9.81529518, 7.43277877, 1.61044578],
       [9.66299965, 7.2814351,  1.67435123],
       [8.78476469, 6.84823437, 0.35153703],
       [8.5316005,  5.30859857, 2.32816275],
       [9.5936181,  6.32616886, 1.05361311]])


df = pd.DataFrame(PE)
df.columns=['Entropy1', "Entropy2", "Entropy3"]


Name= []

for i in range(len(df)):
    if i == len(df)-1:
        Name.append("croissant")
    elif i == len(df)-2:
        Name.append("cylinder")
    elif i == len(df)-3:
        Name.append("sphere")
    elif i == len(df)-4:
        Name.append("Peanut")
    else:
        Name.append('Brain')

df["Names"]=Name

fig = px.scatter_3d(df, x='Entropy1', y='Entropy2', z='Entropy3', color=Name)
fig.show()


clustering = AffinityPropagation(random_state=5).fit(PE)
df["Label"] = clustering.labels_


fig= px.scatter_3d(df, x="Entropy1", y="Entropy2", z="Entropy3", color="Label")
fig.show()

fig= px.scatter(df, x="Entropy2", y="Entropy3", color="Label")
fig.show()


