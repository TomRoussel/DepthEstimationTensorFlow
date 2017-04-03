import numpy as np
from math import floor
import h5py

class ESAT_Data(object):
	"""
		Object that handles loading ESAT data from an HDF file
		Is iterable
	"""
	def __init__(self, hdfFn, config):
		self.hdfFile = h5py.File(hdfFn, 'r')
		self.config = config
		self.batchAm = int(floor(self.hdfFile["depth"]["depth_data"].shape[2]/config["batchSize"]))

		print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))		

	def __getitem__(self, key):
		if key >= self.batchAm:
			raise IndexError
		else:
			startI = key*self.config["batchSize"]
			endI = (key+1)*self.config["batchSize"]
			rgb = np.ones((self.config["batchSize"],self.config["H"],self.config["W"], 3))
			depth = np.ones((self.config["batchSize"],self.config["HOut"],self.config["WOut"]))

			rgb[:,:,:,:] = np.rollaxis(self.hdfFile["rgb"]["rgb_data"][:,:,:,startI:endI],3)
			depth[:,:,:] = np.rollaxis(self.hdfFile["depth"]["depth_data"][:,:,startI:endI],2)
			return rgb, depth

	def __len__(self):
		return self.batchAm
		
