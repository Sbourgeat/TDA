import Entropy
import pandas as pd
import pyvista as pv
import ipywidgets 
import igraph
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import velour


arr = ['859_female.ply',
 '26_male.ply',
 '517_female.ply',
 '850_male.ply',
 '439_female.ply',
 '705_female.ply',
 '195_female.ply',
 '176_female.ply',
 '439_male.ply',
 '59_female.ply',
 '787_male.ply',
 '91_male.ply',
 '900_male.ply',
 '359_female.ply',
 '853_female.ply',
 '359_male.ply',
 '796_male.ply',
 '705_male.ply',
 '176_male.ply',
 '595_male.ply',
 '321_female.ply',
 '26_female.ply',
 '365_female.ply',
 '900_female.ply',
 '853_male.ply',
 '362_male.ply',
 '59_male.ply',
 '321_male.ply',
 '859_male.ply',
 '365_male.ply',
 '739_male.ply',
 '395_female.ply',
 '517_male.ply',
 '405_female.ply',
 '439_male_2.ply',
 '796_female.ply',
 '362_female.ply',
 '595_female.ply',
 '304_male.ply',
 '787_female.ply']



A = []
mypath = "/Users/skumar/TDA/Mesh/"
for i in arr:
    #print(mypath + i)
    A.append(mypath + i)
    


Mesh = []


for i in A:
    mesh = pv.read(i)
    x = mesh.points
    N = 500
    x=x[np.random.choice(len(x), size=N, replace=False)]
    #x= x.reshape(1, *x.shape)
    Mesh.append(x)


X=[]
for i in Mesh:
    X.append(i)
    #print(i)


X = np.array(X)



Res = {}

ctn = 0

for i in X:
    entropy_average = Entropy.Entropy.mean_entropy([i],10)
    Res[arr[ctn]]= entropy_average[0].tolist()
    ctn +=1



#print(Res)


df = pd.DataFrame(Res)
#print(df)

df.to_csv("test_Entropy.csv")