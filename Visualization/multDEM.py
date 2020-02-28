"""
    Multiplies elevations of DEM file by mul.
    Example 'python3 multDEM.py sample/temp.tif 5'
    Output should be sample/temp_mul.tif
"""
import sys, os, csv

dem = None
mul = 1
CRS = "EPSG:4326"
if(len(sys.argv) > 1): dem = sys.argv[1]
if(len(sys.argv) > 2): mul = int(sys.argv[2])

os.system("gdal_translate -of xyz " + dem + " " + dem[:-4] + ".csv")

filename = dem[:-4] + ".csv"
newfilename = dem[:-4] + "-new.csv"

with open(filename, 'r') as inp, open(newfilename, 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        split = row[0].split()
        split[2] = str(int(split[2]) * mul)
        row[0] = " ".join(split)
        writer.writerow(row)

os.system("gdal_translate -of GTiff " + newfilename + " " + dem[:-4] + '_mul.tif')
# os.system("gdal_edit.py -a_srs " + CRS + " " + dem[:-4] + '_mul.tif')
os.remove(filename)
os.remove(newfilename)
