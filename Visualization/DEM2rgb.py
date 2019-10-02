"""
    Gets the corresponding rgb image to a input DEM file, using earth engine data.
    Also generates a csv file from the DEM file, using qgis's gdal2xyz function.
"""
outdir = "sample"
DEM = 'sample'
SATELLITE_SR = 'LANDSAT/LC08/C01/T1_SR'

import ee
import os
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

with rasterio.open(outdir + "/" + DEM + ".dem") as dataset:

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
    print(coord)
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
print(region)
dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).map(mask_l8_sr).select(RGB)
image = dataset.reduce('median')
PERCENTILE_SCALE = 100  # Resolution in meters to compute the percentile at
percentiles = image.reduceRegion(ee.Reducer.percentile([0, 100], ['min', 'max']),
                                 geom, PERCENTILE_SCALE).getInfo()
# Extracting the results is annoying because EE prepends the channel name
minVal = [val for key, val in percentiles.items() if 'min' in key]
# splitVal = next(val for key, val in percentiles.items() if 'split' in key)
maxVal = [val for key, val in percentiles.items() if 'max' in key]
print(minVal)
print(maxVal)
minn = np.amax(np.array(minVal))
mymin = [minn, minn, minn]
maxx = np.amin(np.array(maxVal))
mymax = [maxx, maxx, maxx]
print(mymin)
print(mymax)
NEWRGB = ['B4_median', 'B3_median', 'B2_median']
reduction = image.visualize(bands=NEWRGB,
                            min=mymin, # reverse since bands are given in the other way (b2,b3,4b)
                            max=mymax,
                            gamma=1)

path = reduction.getDownloadUrl({
    'scale': 30,
    'crs': 'EPSG:4326',
    'region': region
})
print(path)
file = re.search("docid=.*&", path).group()[:-1][6:]
print(file)
urllib.request.urlretrieve(path, file + ".zip")

with zipfile.ZipFile(file + ".zip", 'r') as zip_ref:
    zip_ref.extractall(outdir)

for f in glob.glob(outdir + "/*.tfw"):
    os.remove(f)
os.remove(file + ".zip")

red    = Image.open(outdir + '/' + file + '.vis-red.tif')
green  = Image.open(outdir + '/' + file + '.vis-green.tif')
blue   = Image.open(outdir + '/' + file + '.vis-blue.tif')

for f in glob.glob(outdir + "/*.tif"):
    os.remove(f)

rgb = Image.merge("RGB",(red,green,blue))
rgb.save(outdir + "/" + DEM + '.tif')

os.system("gdal2xyz.py -band 1 -csv " + outdir + "/" + DEM + ".dem " + outdir + "/" + DEM + ".csv")

for f in glob.glob(outdir + "/*.xml"):
    os.remove(f)
