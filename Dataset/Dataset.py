"""
Splits all DEM.tif files in the DataDEM folder to smaller 1024x1024 files.
Removes files that have only 0 values (sea).
"""
import os, sys, gdal
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import vstack
import rasterio
SCALE = 10
DATA_SIZE = 256
tile_size_x = 512
def removeZeroTiles(filename, size = (1024,1024)):
    delete = False
    with rasterio.open(filename, 'r') as ds:
        arr = np.array(ds.read())  # read all raster values
        if(np.amax(arr) == 0):
            delete = True
    if(delete):
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros.")
    return delete


path = 'DataDEM/'
out_path = 'preSplitDEM/'
os.system("mkdir " + out_path)
k = 0
CRS = "EPSG:4326"


SIZE = tile_size_x
tile_size_y = tile_size_x
output_filename = 'tile_'
for filename in os.listdir(path):
    dem = filename
    ds = gdal.Open(path + dem)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + path + str(dem) + " " + str(out_path) + str(output_filename) + str(k) + ".tif"
            out = os.system(com_string)
            if(out == 0):
                ret = removeZeroTiles(str(out_path) + str(output_filename) + str(k) + ".tif")
                if(not ret): k += 1

li = os.listdir('preSplitDEM/') # dir is your directory path
allfiles = len(li)
for i in range(allfiles):
    filename = "tile_" + str(i) + ".tif"
    if(i == 0):
        cc = "python3 reSplit.py " + "preSplitDEM/" + filename + " " + str(DATA_SIZE) + " " + str(SCALE)
        os.system(cc)
        print(cc)
    else:
        li = os.listdir('DEM_255/') # dir is your directory path
        number_files = len(li)
        cc = "python3 reSplit.py " + "preSplitDEM/" + filename + " " + str(DATA_SIZE) + " " + str(SCALE) + " " + str(number_files)
        os.system(cc)

os.system("python3 -W ignore TrainingArrays.py")
print("Created Dataset.npz it contains " + str(number_files) + " pairs of Images-DEM.")
print(cc)
li = os.listdir('DEM_255/') # dir is your directory path
number_files = len(li)
print()
print()
print("Done dataset Files Produced: " + str(number_files))
