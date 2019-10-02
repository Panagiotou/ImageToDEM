"""
    Plots a surface plot using matplotlib's plot_surface function.
    Needed files are:
        * csv file corresponding to a valid DEM file
        * rgb tif file corresponding to a valid DEM file for coloring (optional)
        if not given, a heatmap is ploted instead (coloring = False)
"""
outdir = "sample"
DEM = 'sample'
coloring = True
percentageUsed = 0.01 #Percentage of csv file lines used (higher accuracy)

import matplotlib as mpl
import rasterio
from PIL import Image
import csv
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

filename = outdir + "/" + DEM + '.csv'

i = 0
X = []
Y = []
Z = []
with open(filename) as f:
  reader = csv.reader(f)
  for row in reader:
    if(i%int(1/percentageUsed) == 0):
        if(float(row[2]) > 0):
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))
    i+=1

def getcolors(x,y):
    c = []
    im = Image.open(outdir + "/" + DEM + ".tif")
    xsize = im.size[0] -1
    ysize = im.size[1] -1
    rgb_im = im.convert('RGB')
    Xmin = np.amin(x)
    Xmax = np.amax(x)
    Xd = Xmax-Xmin
    Ymin = np.amin(y)
    Ymax = np.amax(y)
    Yd = Ymax-Ymin
    for i in range(len(x)):
        xc = x[i] - Xmin
        yc = y[i] - Ymin
        xp = int(xc/Xd * xsize)
        yp = int(yc/Yd * ysize)
        r, g, b = rgb_im.getpixel((xp, yp))
        c.append((r,g,b))
    return np.array(c)

x = np.array(X)
y = np.array(Y)
z = np.array(Z)

V = getcolors(x,y)/255.0
cm = LinearSegmentedColormap.from_list('my_cm', V, N=100)

# 2D grid construction
spline = sp.interpolate.Rbf(x,y,z,function='thin-plate')
xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
X, Y = np.meshgrid(xi, yi)
# interpolation
Z = spline(X,Y)

fig = plt.figure()
ax = Axes3D(fig)
if(coloring):
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm,linewidth=0, antialiased=True)
else:
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=True)
plt.show()
