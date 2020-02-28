"""
	Loads a Dataset of rgb satellite images-dems and shows first 10 samples
	Example 'python3 LoadData.py' a TestSet.npz file is needed for the data
	Output 2 matplotlib plots of input-output data
"""

import numpy as np
from numpy import load
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
SIZE = 256

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
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = load_real_samples('TestSet.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
dataset[0] = dataset[0][:10]
dataset[1] = dataset[1][:10]
SUBSIZE = int(np.sqrt(len(dataset[0]))) + 1
FIGSIZE = 7
plt.figure(figsize=(FIGSIZE, FIGSIZE))

for i in range(len(dataset[0])):
  rj_inp, rj_re = [dataset[0][i], dataset[1][i]]
  plt.subplot(SUBSIZE, SUBSIZE, i+1)
  plt.imshow((1 + rj_inp)/2)
  plt.axis('off')
plt.figure(figsize=(FIGSIZE, FIGSIZE))
for i in range(len(dataset[0])):
  rj_inp, rj_re = [dataset[0][i], dataset[1][i]]
  rj_re = rj_re[:,:,0]
  plt.subplot(SUBSIZE, SUBSIZE, i+1)
  plt.imshow((1 + rj_re)/2,cmap='gray',norm=NoNorm())
  plt.axis('off')
plt.show()
