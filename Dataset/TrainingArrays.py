"""
	Saves input-output dataset in *.npz format
    Example 'python3 TrainingArrays.py Dataset.npz'
	Output Dataset.npz

	*Expects folders Image and DEM_255 from executing Dataset.py*
"""
SIZE = 256
# dataset path
inPath = 'Image/jpg/'
outPath = 'DEM_255/'
# load, split and scale the maps dataset ready for training
import os, sys
import numpy as np
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import rasterio

# load all images in a directory into memory
def load_out_images(path, path1=None, size=(SIZE,SIZE)):
	tar_list = list()
	# enumerate filenames in directory, assume all are images
	if(path1):
		li = os.listdir(path1) # dir is your directory path
		number_files = len(li)
	else:
		li = os.listdir(path) # dir is your directory path
		number_files = len(li)
	for i in range(number_files):
		filename = 'tile_' + str(i) + '_255.tif'
		with rasterio.open(path + filename, 'r') as ds:
			arr = np.array(ds.read())  # read all raster values
			arr = arr.reshape((SIZE,SIZE,1))
			tar_list.append(arr)
	return asarray(tar_list)

# load all images in a directory into memory
def load_in_images(path, size=(SIZE,SIZE)):
	tar_list = list()
	# enumerate filenames in directory, assume all are images
	li = os.listdir(path) # dir is your directory path
	number_files = len(li)
	for i in range(number_files):
		filename = 'tile_' + str(i) + '.jpg'
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		tar_list.append(pixels)
	return asarray(tar_list)


# load dataset
in_images = load_in_images(inPath)
print('Loaded Input Images: ', in_images.shape)
out_images = load_out_images(outPath, inPath)
print('Loaded Output Images: ', out_images.shape)
# save as compressed numpy array
filename = 'temp.npz'
if(len(sys.argv) > 1):
	filename = sys.argv[1]
savez_compressed(filename, in_images, out_images)
print('Saved dataset: ', filename)
