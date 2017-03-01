# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-01 16:51:25
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN
import h5py
from math import floor

# TODO: Make argument parser

dataFn = "/esat/citrine/tmp/troussel/IROS/kinect/StijnKinectDataNP/1.hdf5" 
configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

rootFolder = "/usr/data/tmp/troussel/IROS/depth_estim/3/"
summaryLoc = "%s/summary" % rootFolder
weightsLoc = "%scheckpoint" % rootFolder

def prepare_data(fn, batchSize, maxFraction = 1):
	"""
		@fn: filename containing gt data
		@batchSize: size of each batch
		@maxFraction: should be between 0 and 1. Determines how much data is used for training
	"""
	assert 0 <= maxFraction <= 1, "Training fraction is not valid, should be between 0 and 1"

	dfile = h5py.File(fn)
	imAmount = dfile["depth"]["depth_data"].shape[2]
	batchAmount = imAmount//batchSize
	maxBatches = int(floor(batchAmount*maxFraction))
	print("Amount of batches: %d" % maxBatches)
	# Loop over all batches
	for x in xrange(maxBatches):
		idStart = x*batchSize; idEnd = (x+1)*batchSize;
		rgb = np.asarray(dfile["rgb"]["rgb_data"][:,:,:,idStart:idEnd])
		depth = np.asarray(dfile["depth"]["depth_data"][:,:,idStart:idEnd])
	
		rgb = np.transpose(rgb, (3,0,1,2)) 
		depth = np.transpose(depth,(2,0,1)) 
		yield (rgb, depth)

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
	dataGenerator = prepare_data(dataFn, network.config["batchSize"], maxFraction = 0.8)
	# Train network
	network.train(dataGenerator, loadChkpt = True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()
