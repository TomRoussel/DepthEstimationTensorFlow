import numpy as np
from time import time
import scipy.misc
from math import floor
import PIL.Image as im

class NYU_Data(object):
	"""
		Object that handles loading NYU data from an HDF file
		Is iterable
	"""
	def __init__(self, rootDataFolder, hdfFile, config):
		self.rootDataFolder = rootDataFolder
		self.hdfFile = hdfFile
		self.config = config
		self.batchAm = int(floor(self.hdfFile["depth"]["depth_labels"].shape[0]/config["batchSize"]))

		print("File contains %d batches using a batchsize of %d" % (self.batchAm, self.config["batchSize"]))		

	def __getitem__(self, key):
		if key >= self.batchAm:
			raise IndexError
		else:
			batch = range(key*self.config["batchSize"],(key+1)*self.config["batchSize"])
			# print("Loading batch %d" % key)
			
			rgb, gtDepth = self.load_batch(batch) 
			# print("Batch loaded")
			return rgb, gtDepth

	def __len__(self):
		return self.batchAm

	def load_batch(self, batch):
		rgb = np.ones((self.config["batchSize"],self.config["H"],self.config["W"], 3))
		depth = np.ones((self.config["batchSize"],self.config["HOut"],self.config["WOut"]))

		i = 0
		start1 = time()
		for entry in batch:
            # Get rgb file
			num = self.hdfFile["depth"]["depth_folder_id"][int(entry)]
			name = self.hdfFile["depth"]["depth_labels"][int(entry)]
			match = "%s/imgs_%d/%s.jpeg" % (self.rootDataFolder, num, name)
			f = open(match, 'rb')
			pilIM = im.open(f)
			pilIm2 = pilIM.copy() #PIL bug workaround
			f.close()
			rgb[i,:,:,:] = np.asarray(pilIM)
			pilIM.close()
			# rgb[i,:,:,:] = scipy.misc.imread(match, mode = "RGB")
			start4 = end3 = time()
			# Get GT depth info
			start5 = end4 = time()
			end4 = time()

			i += 1
			# print("Reading ID: %f, reading label: %f, reading image: %f, reading depth: %f" % (end1-start1,end2-start2,end3-start3,end4-start4))

		start2 = end1 = time()
		depth[:,:,:] = np.swapaxes(np.swapaxes(self.hdfFile["depth"]["depth_data"][:,:,sorted([int(x) for x in batch])],0,2),1,2)
		end2 = time()
		# print("Reading RGB data: %f reading depth: %f" % (end1-start1,end2-start2))
		return rgb, depth
