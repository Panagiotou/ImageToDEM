"""
Generates a dataset of DEM files and their corresponding RGB files.
Inputs:
    * DEM.tif file
    * Pixel size (integer) of the output squares
    * Scale of the RGB file downloaded via Earth Engine
"""
import os, sys, gdal
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from numpy import asarray
from numpy import vstack
import rasterio

dem = None
newW = None
newH = None
scale = None
CRS = "EPSG:4326"

dem = sys.argv[1]
tile_size_x = int(sys.argv[2])
SIZE = tile_size_x
tile_size_y = tile_size_x
scale = int(sys.argv[3])

out_path = 'DEM/'
output_filename = 'tile_'

os.system("mkdir " + out_path)

ds = gdal.Open(dem)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

k = 0
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(dem) + " " + str(out_path) + str(output_filename) + str(k) + ".tif"
        out = os.system(com_string)
        if(out == 0): k += 1
print()
print()
print("Input DEM successfully split into " + str(k) + " DEMs of size " + str(tile_size_x) + "x" + str(tile_size_x))
print("Now downloading RGB tif file...")

os.system("python3 getRGB.py " + str(dem) + " " + str(scale))

im = Image.open(dem)
width, height = im.size
newW = int(width)
newH = int(height)

dem = dem[:-4] + "rgb.tif"
out_path = 'Image/'
output_filename = 'tile_'
os.system("mkdir " + out_path)
os.system("mkdir " + out_path + "tif/")
os.system("mkdir " + out_path + "jpg/")
out_path = 'Image/tif/'

def load_out_images(path, size=(SIZE,SIZE)):
	tar_list = list()
	# enumerate filenames in directory, assume all are images

	li = os.listdir(path) # dir is your directory path
	number_files = len(li)
	for i in range(number_files):
		filename = 'tile_' + str(i) + '.tif'
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		# tar_list.append(pixels)
		with rasterio.open(path + filename, 'r') as ds:
			arr = np.array(ds.read())  # read all raster values
			arr = arr.reshape((SIZE,SIZE,1))
			tar_list.append(arr)
	return asarray(tar_list)


ds = gdal.Open(dem)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
k = 0
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(dem) + " " + str(out_path) + str(output_filename) + str(k) + ".tif"
        out = os.system(com_string)
        if(out == 0): k += 1

for l in range(k):
    jpg = "gdal_translate Image/tif/tile_" + str(l) + ".tif" + " Image/jpg/tile_" + str(l) + ".jpg"
    os.system(jpg)
print("Input RGB successfully split into " + str(k) + " RGB files of size " + str(tile_size_x) + "x" + str(tile_size_x))

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

os.system("python3 TrainingArrays.py")

print("Created Dataset.npz")
