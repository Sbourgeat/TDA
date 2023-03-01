import pyvista as pv








import os

arr = os.listdir()


I=0
A = []
mypath = "/Users/skumar/TDA/Mesh/"
for i in arr:
    if I == len(arr):
        break
    else:
        print(mypath + i)
        A.append(mypath + i)
    I+=1

print(len(A))

#for i in A:
#    mesh = pv.read(i)
#    mesh.plot()

