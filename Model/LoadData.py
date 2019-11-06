import numpy as np
from numpy import load
import rasterio
import matplotlib.pyplot as plt

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

dataset = load_real_samples('../Dataset/Dataset.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
dataset[0] = dataset[0][:10]
dataset[1] = dataset[1][:10]
plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = [dataset[0][i], dataset[1][i]]
  print(rj_inp)
  plt.subplot(2, 2, i+1)
  plt.imshow((1 + rj_inp)/2)
  plt.axis('off')
plt.show()
