"""
    Downloads the corresponding BANDS to a input DEM file, using Google earth engine data.
    Example 'python3 DEM2bands.py sample/temp.tif'
    Output is a numpy array containing all the bands

    *Note, Image is resized to match input DEM width, height*
    Final of the output array is (width, height, #BANDS)
"""
import os, sys
DEM = sys.argv[1]
DEMorig = DEM
tempdir = "tempdir"
os.system("mkdir " + tempdir)
if(os.path.isfile(tempdir + "/" + "temp.tif")):
    os.remove(tempdir + "/" + "temp.tif")

os.system("gdalwarp -ot Float32 -q " + DEM + " " + tempdir + "/" + "temp.tif")
DEM = "temp"

outdir = tempdir
SATELLITE_SR = "COPERNICUS/S2_SR"
BANDS = [['B4', 'B3', 'B2'], ['B7', 'B6', 'B5'], ['B11', 'B8A', 'B8']]
NEWBANDS = [['B4_median', 'B3_median', 'B2_median'], ['B7_median', 'B6_median', 'B5_median'], ['B11_median', 'B8A_median', 'B8_median']]
SCALE = 50
PERCENTILE_SCALE = 50  # Resolution in meters to compute the percentile at

if(len(sys.argv) > 2):
    SCALE = int(sys.argv[2])
    PERCENTILE_SCALE = int(sys.argv[2])

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

with rasterio.open(outdir + "/" + DEM + ".tif") as dataset:

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
        print("Geometry selected:")
        print(geom)

Xmin = geom['coordinates'][0][0][0]
Xmax = geom['coordinates'][0][0][0]
Ymin = geom['coordinates'][0][0][1]
Ymax = geom['coordinates'][0][0][1]


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
print("Region selected from Geometry:")
print(region)
allbands = []
for w in range(len(BANDS)):
    dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).select(BANDS[w])
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
    reduction = image.visualize(bands=NEWBANDS[w],
                                min=mymin, # reverse since bands are given in the other way (b2,b3,4b)
                                max=mymax,
                                gamma=1)
    path = image.getDownloadUrl({
        'scale': SCALE,
        'crs': 'EPSG:4326',
        'maxPixels': 1e20,
        'region': region,
        'bestEffort': True
    })
    print(path)
    file = re.search("docid=.*&", path).group()[:-1][6:]
    print(file)
    urllib.request.urlretrieve(path, outdir + "/" + file + ".zip")

    with zipfile.ZipFile(outdir + "/" + file + ".zip", 'r') as zip_ref:
        zip_ref.extractall(outdir)

    for f in glob.glob(outdir + "/*.tfw"):
        os.remove(f)
    os.remove(outdir + "/" + file + ".zip")

    for f in glob.glob(outdir + "/*.tfw"):
        os.remove(f)

    red    = Image.open(outdir + '/' + file + '.' + NEWBANDS[w][0] + '.tif')
    green  = Image.open(outdir + '/' + file + '.' + NEWBANDS[w][1] +'.tif')
    blue   = Image.open(outdir + '/' + file + '.' + NEWBANDS[w][2] +'.tif')

    im = Image.open(DEMorig)
    width, height = im.size
    red = red.resize((int(width), int(height)),Image.ANTIALIAS)
    green = green.resize((int(width), int(height)),Image.ANTIALIAS)
    blue = blue.resize((int(width), int(height)),Image.ANTIALIAS)

    bands = np.dstack((red,green,blue))
    allbands.append(bands)
    os.system('rm -rf ' + tempdir)
    os.system("mkdir " + tempdir)
os.system('rm -rf ' + tempdir)
print()
outbands = np.dstack(tuple(allbands))
print("Downloaded bands", BANDS, "into numpy array of shape:")
print(outbands.shape)
