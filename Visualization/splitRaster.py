"""
    Split DEM into smaller tiles of size (tile_size_x, tile_size_y)
    Example 'python3 splitRaster.py'
    Output should be in the 'out_path' directory containing all the tiles
"""
import os, gdal

in_path = 'sample/'
input_filename = 'temp.tif'

out_path = 'sample/split/'
output_filename = 'tile_'

if(not os.path.isdir(out_path)):
    os.system("mkdir " + out_path)

tile_size_x = 100
tile_size_y = 100

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
