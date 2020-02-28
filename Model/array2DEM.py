"""
	Saves a (SIZE, SIZE, 1) numpy array as a DEM file
	Example 'python3 array2DEM.py' with TestSet.npz array as input a sample DEM is also needed
	Output sample/array.tif
"""

import os
import numpy as np
from numpy import load
import rasterio

SIZE = 256
def saveDEM(outname, temp, arr):
    os.system("cp " + str(temp) + " " + str(outname))
    with rasterio.open(outname, 'r+') as ds:
    	ds.write(arr.reshape((1,SIZE,SIZE)))

def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
def denormalize(arr, oldMin, oldMax, newMin, newMax):
	minn = oldMin
	maxx = oldMax
	arr = (newMax - newMin) * (arr - minn)/(maxx - minn) + newMin
	return arr.astype('int16', copy=False)

# Load a sample DEM to have a reference
tempDEM = 'sample/temp.tif'
outName = 'sample/array.tif'
dataset = load_real_samples('TestSet.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# Load a sample array
tif = dataset[1][0]
tif = (1+tif)/2
tif = denormalize(tif, 0, 1, 0, 255)



saveDEM(outName, tempDEM, tif)
