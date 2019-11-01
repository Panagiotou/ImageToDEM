import numpy as np
from numpy import load
import rasterio

SIZE = 256
tempDEM = 'temp.tif'

def normalize(arr, oldMin, oldMax, newMin, newMax):
	minn = oldMin
	maxx = oldMax
	arr = (newMax - newMin) * (arr - minn)/(maxx - minn) + newMin
	return arr
def denormalize(arr, oldMin, oldMax, newMin, newMax):
	minn = oldMin
	maxx = oldMax
	arr = (newMax - newMin) * (arr - minn)/(maxx - minn) + newMin
	return arr.astype(int, copy=False)

# load the face dataset
data = load('../Dataset/Dataset.npz')
in_images, out_images = data['arr_0'], data['arr_1']
print('Loaded: ', in_images.shape, out_images.shape)

for arr in out_images:
	print(arr[0][0])
	arr = (arr - 127.5) / 127.5
	print(arr[0][0])
	arr = (arr + 1) / 2.0
	print(arr[0][0])
	arr = denormalize(arr, 0, 1, 0, 255)
	print(arr[0][0])


def saveDEM(temp, arr):
    with rasterio.open(temp, 'r+') as ds:
    	ds.write(arr.reshape((1,SIZE,SIZE)))

saveDEM(tempDEM, out_images[0])
