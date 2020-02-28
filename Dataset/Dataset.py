"""
    -preSplits all *.tif files in the DataDEM folder to smaller 1024x1024 files.
    -Removes files that have only 0 values (sea).
    -Downloads RGB Images for every split 1024x1024 tile.
    -Splits DEMs and Images again to get 256x256 tiles.
    -DEMs are normalized to 0-255
    
    Example 'python3 Dataset.py'
    Output a Dataset.npz file containing all in-out data is saved periodically
"""
import os, sys, gdal
import numpy as np
from numpy import asarray
from numpy import vstack
import rasterio
import re
from os import listdir
from os.path import isfile, join
import ee
import time
import requests
import rasterio.features
import rasterio.warp
from requests.auth import HTTPBasicAuth
import urllib
import zipfile
import glob
from PIL import Image
import time


DatasetName = "Dataset.npz"

path = 'DataDEM/'
out_path = 'preSplitDEM/'

ee.Initialize()

SCALE = 50
DATA_SIZE = 256
tile_size_x = 512
SATELLITE_SR = "COPERNICUS/S2_SR"
def removeZeroTiles(filename):
    delete = False
    with rasterio.open(filename, 'r') as ds:
        arr = np.array(ds.read())  # read all raster values
        if(np.amax(arr) == 0):
            delete = True
    if(delete):
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros.")
    return delete

def removeZeroTilesRGB(filename):
    delete = False
    filenameN = re.search(r'tif/(.*?)\.', filename).group(1)
    exists = "DEM/" + filenameN + ".tif"
    print(exists)
    if(not os.path.isfile(exists)):
        delete = True
    if(delete):
        os.remove(filename)
        print("File:" + filename + " removed, it contained only zeros. (rgb file)")
    return delete

def load_out_images(path, size=(DATA_SIZE, DATA_SIZE)):
	tar_list = list()
	# enumerate filenames in directory, assume all are images

	li = os.listdir(path) # dir is your directory path
	number_files = len(li)
	for i in range(number_files):
		filename = 'tile_' + str(i) + '.tif'
		with rasterio.open(path + filename, 'r') as ds:
			arr = np.array(ds.read())  # read all raster values
			arr = arr.reshape((SIZE,SIZE,1))
			tar_list.append(arr)
	return asarray(tar_list)



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
number_files = 0
CRS = "EPSG:4326"

out_path_re = 'DEM/'
output_filename_re = 'tile_'
os.system("mkdir " + out_path_re)

out_path_rergb = 'Image/'
output_filename_rergb = 'tile_'
os.system("mkdir " + out_path_rergb)
os.system("mkdir " + out_path_rergb + "tif/")
os.system("mkdir " + out_path_rergb + "jpg/")
os.system("mkdir DEM_255")
out_path_rergb = 'Image/tif/'
tempdir = "tempdir"

out_path1 = 'preSplitImages/'
os.system("mkdir " + out_path1)
dem_times = []
for ai in range(allfiles):
    try:
        start = time.time()
        filename = "tile_" + str(ai) + ".tif"
        dem = "preSplitDEM/" + filename
        tile_size_x = int(DATA_SIZE)
        SIZE = tile_size_x
        tile_size_y = tile_size_x
        scale = int(SCALE)
        lastk = number_files

        ds = gdal.Open(dem)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        if(ai > 0 ):
            print ('Estimated Time left for remaining {} DEMs is {} sec = {} hours\n'.format(allfiles - ai, (allfiles - ai) * sum(dem_times) / len(dem_times), (allfiles - ai) * sum(dem_times) / len(dem_times)/3600))
        print("Now downloading RGB tif file...")

        DEMorig = dem
        preSplitImages = re.search(r'/(.*?)\.', DEMorig).group(1)

        os.system("mkdir " + tempdir)
        if(os.path.isfile(tempdir + "/" + "temp.tif")):
            os.remove(tempdir + "/" + "temp.tif")

        os.system("gdalwarp -s_srs " + CRS + " -ot Float32 -q " + DEMorig + " " + tempdir + "/" + "temp.tif")
        DEM = "temp"


        outdir = "Image"
        PERCENTILE_SCALE = SCALE  # Resolution in meters to compute the percentile a

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

        region = '[[{}, {}], [{}, {}], [{}, {}], [{}, {}]]'.format(Xmin, Ymax, Xmax, Ymax, Xmax, Ymin, Xmin, Ymin)



        # dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).map(mask_l8_sr).select(RGB)
        # dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).select(RGB)
        dataset = ee.ImageCollection(SATELLITE_SR).filterBounds(geom).select(RGB).filter(ee.Filter.calendarRange(2018,2019,'year')).filter(ee.Filter.calendarRange(6,8,'month'));

        image = dataset.reduce('median')
        percentiles = image.reduceRegion(ee.Reducer.percentile([0, 100], ['min', 'max']),
                                         geom, PERCENTILE_SCALE, bestEffort=True).getInfo()
        # Extracting the results is annoying because EE prepends the channel name
        mymin = [percentiles['B4_median_min'], percentiles['B3_median_min'], percentiles['B2_median_min']]
        mymax = [percentiles['B4_median_max'], percentiles['B3_median_max'], percentiles['B2_median_max']]

        minn = np.amax(np.array(mymin))
        maxx = np.amin(np.array(mymax))
        NEWRGB = ['B4_median', 'B3_median', 'B2_median']
        reduction = image.visualize(bands=NEWRGB,
                                    min=[minn, minn, minn],
                                    max=[maxx, maxx, maxx],
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

        try:
          with zipfile.ZipFile(tempdir + "/" + file + ".zip", 'r') as zip_ref:
              zip_ref.extractall(tempdir)
        except:
          print("Bad Zip file, continuing...")
          os.system('rm -rf ' + tempdir)
          continue



        red    = Image.open(tempdir + '/' + file + '.vis-red.tif')
        green  = Image.open(tempdir + '/' + file + '.vis-green.tif')
        blue   = Image.open(tempdir + '/' + file + '.vis-blue.tif')

        im = Image.open(dem)
        width, height = im.size
        rgb = Image.merge("RGB",(red,green,blue))
        rgb = rgb.resize((int(width), int(height)),Image.ANTIALIAS)


        rgb.save(out_path1 + preSplitImages + 'rgb.tif')
        rgb.save(out_path1 + preSplitImages + 'rgb.jpg')

        for f in glob.glob(outdir + "/*.xml"):
            os.remove(f)

        os.system('rm -rf ' + tempdir)

        demrgb = "preSplitImages/" + re.search(r'/(.*?)\.', dem).group(1) + "rgb.tif"

        if(lastk != 0):
            k = lastk
        else:
            k = 0

        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(dem) + " " + str(out_path_re) + str(output_filename_re) + str(k) + ".tif"
                out = os.system(com_string)
                if(out == 0):
                    ret = removeZeroTiles(str(out_path_re) + str(output_filename_re) + str(k) + ".tif")
                    if(not ret):
                        com_string = "gdal_translate -of GTIFF -epo -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(demrgb) + " " + str(out_path_rergb) + str(output_filename_rergb) + str(k) + ".tif"
                        out = os.system(com_string)
                        k += 1
        print()
        print()
        li = os.listdir('preSplitDEM/') # dir is your directory path
        allfiles = len(li)
        print("Input DEM number " + str(ai) + "/" + str(allfiles) + " successfully split into " + str(k-lastk) + " DEMs of size " + str(tile_size_x) + "x" + str(tile_size_x))

        for l in range(lastk, k):
            jpg = "gdal_translate Image/tif/tile_" + str(l) + ".tif" + " Image/jpg/tile_" + str(l) + ".jpg"
            os.system(jpg)
        print("Input RGB successfully split into " + str(k-lastk) + " RGB files of size " + str(tile_size_x) + "x" + str(tile_size_x))

        out_images = load_out_images('DEM/')

        Tmin = 1000000000
        Tmax = -1000000000
        for arr in out_images:
        	if(np.amin(arr) < Tmin):
        		Tmin = np.amin(arr)
        	if(np.amax(arr) > Tmax):
        		Tmax = np.amax(arr)

        li = os.listdir("DEM/") # dir is your directory path
        number_files_1 = len(li)
        for i in range(number_files_1):
        	filename = 'tile_' + str(i)
        	os.system("gdal_translate -scale " + str(Tmin) + " " + str(Tmax) + " 0 255 DEM/" + filename + ".tif DEM_255/" +filename + "_255.tif")

        dem_times.append(time.time()-start)
        li = os.listdir('DEM/') # dir is your directory path
        number_files = len(li)
        if(ai == 0 or ai % 10 == 0 or ai == allfiles-1):
            os.system("python3 -W ignore TrainingArrays.py " + DatasetName)
            print("Created Dataset.npz it contains " + str(number_files) + " pairs of Images-DEM.")
            print()
            print()
            print("Done dataset Files Produced: " + str(number_files))
    except:
        print("something went wrong.")
        continue
