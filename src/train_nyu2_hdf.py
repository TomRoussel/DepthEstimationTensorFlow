# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-03 15:05:44
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN
from util.util import loadmat
from math import floor
from glob import glob
from random import shuffle
import os.path
import scipy.misc
import h5py
from time import time

# TODO: Make argument parser

trainSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/train"
validateSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/validate"
testSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/test"

configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

depthFile = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_train2.hdf5"
rootData = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/"

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/NYU_proper_random/"
summaryLoc = "%ssummary" % rootFolder
weightsLoc = "%scheckpoint/" % rootFolder

def load_batch(batch, hdfFile, rgbFiles, config):
	rgb = np.ones((config["batchSize"],config["H"],config["W"], 3))
	depth = np.ones((config["batchSize"],config["HOut"],config["WOut"]))

	i = 0
	lastPath = None
	start1 = time()
	for entry in batch:
		# Get rgb file
		num = hdfFile["depth"]["depth_folder_id"][int(entry)]
		name = hdfFile["depth"]["depth_labels"][int(entry)]
		match = "%s/imgs_%d/%s.jpeg" % (rgbFiles, num, name)
		rgb[i,:,:,:] = scipy.misc.imread(match, mode = "RGB")
		start4 = end3 = time()
		# Get GT depth info
		start5 = end4 = time()
		end4 = time()

		i += 1
		# print("Reading ID: %f, reading label: %f, reading image: %f, reading depth: %f" % (end1-start1,end2-start2,end3-start3,end4-start4))

	start2 = end1 = time()
	depth[:,:,:] = np.swapaxes(np.swapaxes(hdfFile["depth"]["depth_data"][:,:,sorted([int(x) for x in batch])],0,2),1,2)
	end2 = time()
	print("Reading RGB data: %f reading depth: %f" % (end1-start1,end2-start2))
	return rgb, depth

def train_data_generator(rootDataFolder, hdfFile, config):
	# Split manifest in batches
	batchAm = int(floor(hdfFile["depth"]["depth_labels"].shape[0]/config["batchSize"]))
	indexes = range(hdfFile["depth"]["depth_labels"].shape[0])
	# Loop over batches
	print("Looping over %d batches" % batchAm)
	for x in xrange(batchAm):
		# Get Batch
		batch = indexes[x*config["batchSize"]:(x+1)*config["batchSize"]]
		# Load data
		print("Loading batch %d" % x)
		rgb, gtDepth = load_batch(batch, hdfFile, rootDataFolder, config) 
		print("Batch loaded")
		# yield data
		yield rgb, gtDepth

def prepare_data(depthFile, rootDataFolder, config):
	"""
	"""
	# assert split[0] + split[1] + split[2] == 1
	hdfF = h5py.File(depthFile, 'r')

	# Prepare generator with training split
	print("Preparing data generator")
	return train_data_generator(rootDataFolder, hdfF, config)


def safe_mkdir(dir):
	"""
		Makes directory but does not throw exception when it already exists
	"""
	import os
	try:
		os.mkdir(dir)
	except OSError as e:
		if e[0] == 17: #Dir already exists
			return
		else:
			raise e

def prepare_dir(summaryLoc, weightsLoc):
	safe_mkdir(rootFolder)
	safe_mkdir(summaryLoc)
	safe_mkdir(weightsLoc)
	
def main():
	# Make network
	prepare_dir(summaryLoc, weightsLoc)
	print("Loading network configuration")
	network = DEN(summaryLoc, weightsLoc, confFileName = configFile, training = True)
	# Prepare training data
	print("Preparing data")
	dataGenerator = prepare_data(depthFile, rootData, network.config)
	# Train network
	print("Starting training")
	network.train(dataGenerator, loadChkpt = False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()
