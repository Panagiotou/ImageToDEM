"""
Generates a dataset of DEM files and their corresponding RGB files.
Inputs:
    * DEM.tif file
    * Pixel size (integer) of the output squares
    * Scale of the RGB file downloaded via Earth Engine
"""
import os, sys, gdal
from PIL import Image

dem = None
newW = None
newH = None
scale = None
CRS = "EPSG:4326"

dem = sys.argv[1]
tile_size_x = int(sys.argv[2])
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
