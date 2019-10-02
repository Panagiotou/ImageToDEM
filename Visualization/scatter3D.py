"""
    Plots a surface plot using matplotlib's scatter3D function.
    Needed files are:
        * csv file corresponding to a valid DEM file
        * rgb tif file corresponding to a valid DEM file for coloring (optional)
        if not given, a heatmap is ploted instead (coloring = False)
"""
outdir = "sample"
DEM = 'sample'
coloring = True
percentageUsed = 0.1 #Percentage of csv file lines used (higher accuracy)


from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
import csv
from matplotlib import cm
from osgeo import gdal
from PIL import Image

filename = outdir + "/" + DEM + '.csv'
i = 0
X = []
Y = []
Z = []
with open(filename) as f:
  reader = csv.reader(f)
  for row in reader:
    # do something here with `row`
    if(i%int(1/percentageUsed) == 0):
        if(float(row[2]) > 0):
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))
    i+=1

x = np.array(X)
y = np.array(Y)
z = np.array(Z)

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

fig = plt.figure()
ax = Axes3D(fig)
cm = getcolors(x,y)/255.0

if coloring:
    ax.scatter3D(x,y,z,c=getcolors(x,y)/255.0, alpha = 1)
else:
    ax.scatter3D(x,y,z,c=z,cmap=plt.cm.jet)

plt.show()
