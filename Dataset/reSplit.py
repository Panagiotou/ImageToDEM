"""
Generates a dataset of DEM files and their corresponding RGB files.
Inputs:
    * DEM.tif file
    * Pixel size (integer) of the output squares
    * Scale of the RGB file downloaded via Earth Engine
"""
import os, sys, gdal
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import vstack
import rasterio
import re

dem = None
newW = None
newH = None
scale = None
lastk = 0
CRS = "EPSG:4326"

def removeZeroTiles(filename):
    delete = False
    with rasterio.open(filename, 'r') as ds:
        arr = np.array(ds.read())  # read all raster values
        if(np.amax(arr) == 0):
            delete = True
    if(delete):
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros.")
    return delete

def removeZeroTilesRGB(filename):
    delete = False
    filenameN = re.search(r'tif/(.*?)\.', filename).group(1)
    exists = "DEM/" + filenameN + ".tif"
    print(exists)
    if(not os.path.isfile(exists)):
        delete = True
    if(delete):
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros. (rgb file)")
    return delete

dem = sys.argv[1]
tile_size_x = int(sys.argv[2])
SIZE = tile_size_x
tile_size_y = tile_size_x
scale = int(sys.argv[3])
if(len(sys.argv) > 4):
    lastk = int(sys.argv[4])

out_path = 'DEM/'
output_filename = 'tile_'

os.system("mkdir " + out_path)
ds = gdal.Open(dem)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize


print("Now downloading RGB tif file...")

os.system("python3 getRGB.py " + str(dem) + " " + str(scale))

im = Image.open(dem)
width, height = im.size
newW = int(width)
newH = int(height)
demrgb = "preSplitImages/" + re.search(r'/(.*?)\.', dem).group(1) + "rgb.tif"
out_pathrgb = 'Image/'
output_filenamergb = 'tile_'
os.system("mkdir " + out_pathrgb)
os.system("mkdir " + out_pathrgb + "tif/")
os.system("mkdir " + out_pathrgb + "jpg/")
out_pathrgb = 'Image/tif/'

if(lastk):
    k = lastk
else:
    k = 0
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(dem) + " " + str(out_path) + str(output_filename) + str(k) + ".tif"
        out = os.system(com_string)
        if(out == 0):
            ret = removeZeroTiles(str(out_path) + str(output_filename) + str(k) + ".tif")
            if(not ret):
                com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(demrgb) + " " + str(out_pathrgb) + str(output_filenamergb) + str(k) + ".tif"
                out = os.system(com_string)
                k += 1
print()
print()
li = os.listdir('preSplitDEM/') # dir is your directory path
allfiles = len(li)
print("Input DEM number " + str(int(lastk/4)) + "/" + str(allfiles) + " successfully split into " + str(k-lastk) + " DEMs of size " + str(tile_size_x) + "x" + str(tile_size_x))


def load_out_images(path, size=(SIZE,SIZE)):
	tar_list = list()
	# enumerate filenames in directory, assume all are images

	li = os.listdir(path) # dir is your directory path
	number_files = len(li)
	for i in range(number_files):
		filename = 'tile_' + str(i) + '.tif'
		with rasterio.open(path + filename, 'r') as ds:
			arr = np.array(ds.read())  # read all raster values
			arr = arr.reshape((SIZE,SIZE,1))
			tar_list.append(arr)
	return asarray(tar_list)


for l in range(lastk, k):
    jpg = "gdal_translate Image/tif/tile_" + str(l) + ".tif" + " Image/jpg/tile_" + str(l) + ".jpg"
    os.system(jpg)
print("Input RGB successfully split into " + str(k-lastk) + " RGB files of size " + str(tile_size_x) + "x" + str(tile_size_x))

out_images = load_out_images('DEM/')
os.system("mkdir DEM_255")
Tmin = 1000000000
Tmax = -1000000000
for arr in out_images:
	if(np.amin(arr) < Tmin):
		Tmin = np.amin(arr)
	if(np.amax(arr) > Tmax):
		Tmax = np.amax(arr)

li = os.listdir("DEM/") # dir is your directory path
number_files = len(li)
for i in range(number_files):
	filename = 'tile_' + str(i)
	os.system("gdal_translate -scale " + str(Tmin) + " " + str(Tmax) + " 0 255 DEM/" + filename + ".tif DEM_255/" +filename + "_255.tif")

li = os.listdir("DEM/") # dir is your directory path
number_files = len(li)
if(number_files % 100 == 0 or number_files < 5):
    os.system("python3 -W ignore TrainingArrays.py")
    print("Created Dataset.npz it contains " + str(number_files) + " pairs of Images-DEM.")
