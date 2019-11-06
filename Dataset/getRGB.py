import os, sys, gdal
from os import listdir
from os.path import isfile, join
import ee
import time
import requests
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
from requests.auth import HTTPBasicAuth
import re
import urllib
import zipfile
import glob, os
from PIL import Image

dem = None
scale = None

dem = sys.argv[1]



"""
    Gets the corresponding rgb image to a input DEM file, using earth engine data.
    Also generates a csv and tif file from the DEM file, using qgis's gdal2xyz
    and gdal_translate function.
"""
import os, sys
DEMorig = sys.argv[1]
preSplitImages = re.search(r'/(.*?)\.', DEMorig).group(1)
tempdir = "tempdir"
os.system("mkdir " + tempdir)
if(os.path.isfile(tempdir + "/" + "temp.tif")):
    os.remove(tempdir + "/" + "temp.tif")

os.system("gdalwarp -ot Float32 -q " + DEMorig + " " + tempdir + "/" + "temp.tif")
DEM = "temp"

outdir = "Image"
SATELLITE_SR = "COPERNICUS/S2_SR"
SCALE = 50
PERCENTILE_SCALE = 50  # Resolution in meters to compute the percentile at

if(len(sys.argv) > 2):
    SCALE = int(sys.argv[2])
    PERCENTILE_SCALE = int(sys.argv[2])



with rasterio.open(tempdir + "/" + DEM + ".tif") as dataset:

    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, 'EPSG:4326', geom, precision=6)
        # Print GeoJSON shapes to stdout.

Xmin = geom['coordinates'][0][0][0]
Xmax = geom['coordinates'][0][0][0]
Ymin = geom['coordinates'][0][0][1]
Ymax = geom['coordinates'][0][0][1]
RGB = ['B4', 'B3', 'B2']
for coord in geom['coordinates'][0]:
    if coord[0] < Xmin:
        Xmin = coord[0]
    elif coord[0] > Xmax:
        Xmax = coord[0]
    if coord[1] < Ymin:
        Ymin = coord[1]
    elif coord[1] > Ymax:
        Ymax = coord[1]

def mask_l8_sr(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) and (qa.bitwiseAnd(clouds_bit_mask).eq(0))
    return image.updateMask(mask)

region = '[[{}, {}], [{}, {}], [{}, {}], [{}, {}]]'.format(Xmin, Ymax, Xmax, Ymax, Xmax, Ymin, Xmin, Ymin)
ee.Initialize()
# dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).map(mask_l8_sr).select(RGB)
# dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).select(RGB)
dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).select(RGB).filter(ee.Filter.calendarRange(2018,2019,'year')).filter(ee.Filter.calendarRange(7,7,'month'));
image = dataset.reduce('median')
percentiles = image.reduceRegion(ee.Reducer.percentile([0, 100], ['min', 'max']),
                                 geom, PERCENTILE_SCALE, bestEffort=True).getInfo()
# Extracting the results is annoying because EE prepends the channel name
minVal = [val for key, val in percentiles.items() if 'min' in key]
# splitVal = next(val for key, val in percentiles.items() if 'split' in key)
maxVal = [val for key, val in percentiles.items() if 'max' in key]
minn = np.amax(np.array(minVal))
mymin = [minn, minn, minn]
maxx = np.amin(np.array(maxVal))
mymax = [maxx, maxx, maxx]
NEWRGB = ['B4_median', 'B3_median', 'B2_median']
reduction = image.visualize(bands=NEWRGB,
                            min=mymin, # reverse since bands are given in the other way (b2,b3,4b)
                            max=mymax,
                            gamma=1)

path = reduction.getDownloadUrl({
    'scale': SCALE,
    'crs': 'EPSG:4326',
    'maxPixels': 1e20,
    'region': region,
    'bestEffort': True
})
file = re.search("docid=.*&", path).group()[:-1][6:]
urllib.request.urlretrieve(path, tempdir + "/" + file + ".zip")

with zipfile.ZipFile(tempdir + "/" + file + ".zip", 'r') as zip_ref:
    zip_ref.extractall(tempdir)

for f in glob.glob(tempdir + "/*.tfw"):
    os.remove(f)
os.remove(tempdir + "/" + file + ".zip")

red    = Image.open(tempdir + '/' + file + '.vis-red.tif')
green  = Image.open(tempdir + '/' + file + '.vis-green.tif')
blue   = Image.open(tempdir + '/' + file + '.vis-blue.tif')

im = Image.open(dem)
width, height = im.size
rgb = Image.merge("RGB",(red,green,blue))
rgb = rgb.resize((int(width), int(height)),Image.ANTIALIAS)

out_path1 = 'preSplitImages/'
os.system("mkdir " + out_path1)


rgb.save(out_path1 + preSplitImages + 'rgb.tif')
rgb.save(out_path1 + preSplitImages + 'rgb.jpg')

for f in glob.glob(outdir + "/*.xml"):
    os.remove(f)

os.system('rm -rf ' + tempdir)
