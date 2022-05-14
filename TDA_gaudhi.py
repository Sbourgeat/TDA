import gudhi
import numpy as np
import matplotlib.pyplot as plt
import velour
import pyvista as pv

mesh = pv.read('Mesh.ply')

#cpos = mesh.plot()

X = mesh.points
X2d = X[0:1000]


plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(X2d)
plotter.show(screenshot="myscreenshot.png")


' Usual Rips complex on X '
st_rips = gudhi.RipsComplex(X2d).create_simplex_tree(max_dimension=2) # create a Rips complex     
diagram_rips = st_rips.persistence()                                # compute the persistence

# plot the persistence diagram
gudhi.plot_persistence_diagram(diagram_rips)                    
plt.title('Persistence diagram of the Rips complex')
plt.show()

filtration_max = 0.5                                             #maximal filtration value for Rips complex
st_Rips = velour.RipsComplex(X2d, filtration_max=filtration_max)   #builds the Rips filtration     

velour.PlotPersistenceDiagram(st_Rips)  
plt.show()

m = 0.1                          #parameter for the DTM
DTM_values = velour.DTM(X2d,X2d,m)   #computes the values of the DTM of parameter m
             
velour.PlotPointCloud(X2d, values = DTM_values) #draws X and the values of DTM
plt.show()

st_alpha = velour.AlphaComplex(X2d)         #creates an alpha-complex

velour.PlotPersistenceDiagram(st_alpha)   #displays the persistence diagram
plt.show()


velour.PlotPersistenceBarcodes(st_alpha, eps=.01)   #displays the persistence barcodes
plt.show()



' DTM with parameter m = 0.05 '
m = 0.05
DTM_values = velour.DTM(X2d,X2d,m)
velour.PlotPointCloud(X2d, values = DTM_values)         
plt.title('Parameter m = '+repr(m))
plt.show()

' DTM with parameter m = 0.3 '
m = 0.3
DTM_values = velour.DTM(X2d,X2d,m)
velour.PlotPointCloud(X2d, values = DTM_values)
plt.title('Parameter m = '+repr(m))
plt.show()