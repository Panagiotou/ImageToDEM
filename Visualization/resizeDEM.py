"""
    Resize DEM width, height by 1/r.
    Example 'python3 resizeDEM.py sample/temp.tif 5'
    Output should be sample/temp_re.tif
"""
import sys, os, csv
from PIL import Image

dem = None
r = 1
CRS = "EPSG:4326"
if(len(sys.argv) > 1): dem = sys.argv[1]
if(len(sys.argv) > 2): r = int(sys.argv[2])

im = Image.open(dem)
width, height = im.size
newW = int(width/r)
newH = int(height/r)
print(newW, newH)
os.system("gdalwarp -of GTiff -ts " + str(newW) + " " + str(newH) + " " + dem + " " + dem[:-4] + "_re.tif")
