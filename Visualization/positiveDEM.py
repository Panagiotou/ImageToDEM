"""
    Removes negative elevations from csv-DEM file.
"""
outdir = "threejs-dem-visualizer/src/textures"
DEM = 'sample'
import csv
import os

JPG = "agri-small-autumn"
TIF = "agri-small-dem"
filename = "sample/" + DEM + '.csv'
newfilename = DEM + 'pos.csv'


with open(filename, 'r') as inp, open(newfilename, 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if(float(row[2]) > 0):
            writer.writerow(row)

os.system("gdal_translate " + newfilename + " " + outdir + "/" + TIF + '.tif')
os.system("cp sample/sample.jpg " + outdir + "/" + JPG + '.jpg')
os.remove(newfilename)
