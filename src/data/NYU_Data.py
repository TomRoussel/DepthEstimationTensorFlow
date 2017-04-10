import numpy as np
from time import time
import scipy.misc
from math import floor
import PIL.Image as im
import h5py

# FIXME: test new implementation
class NYU_Data(object):
	"""
		Object that handles loading NYU data from an HDF file
		Is iterable
		@batchSize: amount of images per batch
		@rootDataFolder: root folder containing the RGB data
		@hdfFn: Path to the .hdf5 file
	"""
	def __init__(self, rootDataFolder, hdfFn, batchSize):
		self.rootDataFolder = rootDataFolder
		self.hdfFile = h5py.File(hdfFn, 'r')
		self.batchSize = batchSize
		self.batchAm = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0]/batchSize))

		print("File contains %d batches using a batchsize of %d" % (self.batchAm, batchSize))		

	def __getitem__(self, key):
		if key >= self.batchAm:
			raise IndexError
		else:
			batch = range(key*self.batchSize,(key+1)*self.batchSize)
			
			rgb, gtDepth = self.load_batch(batch) 
			return rgb, gtDepth

	def __len__(self):
		return self.batchAm

	def load_batch(self, batch):
		# Some convenience constants
		shapeDepth = self.hdfFile["depth"]["depth_data"].shape[1:3]
		
		rgb = None
		depth = np.zeros((self.batchSize,) + shapeDepth)

		i = 0
		for entry in batch:
            # Get rgb file
			num = self.hdfFile["depth"]["depth_folder_id"][int(entry)]
			name = self.hdfFile["depth"]["depth_labels"][int(entry)]
			match = "%s/imgs_%d/%s.jpeg" % (self.rootDataFolder, num, name)
			f = open(match, 'rb')
			pilIM = im.open(f)
			pilIm2 = pilIM.copy() #PIL bug workaround
			f.close()
			imageArray = np.asarray(pilIM)
			# Initialize array if it is not already created
			if not rgb:
				rgb = np.ones((self.batchSize,) + imageArray.shape)
			rgb[i,:,:,:] = imageArray
			pilIM.close()
			# Get GT depth info
			start5 = end4 = time()
			end4 = time()

			i += 1
		
		depth[:,:,:] = np.swapaxes(np.swapaxes(self.hdfFile["depth"]["depth_data"][:,:,sorted([int(x) for x in batch])],0,2),1,2)
		return rgb, depth
