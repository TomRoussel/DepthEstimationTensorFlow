# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-02-20 13:58:27
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN
import h5py

# TODO: Make argument parser

dataFn = "/esat/citrine/tmp/troussel/IROS/kinect/StijnKinectDataNP/1.hdf5" 
configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

summaryLoc = "/usr/data/tmp/troussel/IROS/depth_estim/1"
weightsLoc = "/usr/data/tmp/troussel/IROS/depth_estim/1"

def prepare_data(fn, batchSize):
	dfile = h5py.File(fn)
	imAmount = dfile["depth"]["depth_data"].shape[2]
	batchAmount = floor(batchSize/imAmount)
	# Loop over all batches
	for x in xrange(batchAmount):
		idStart = x*batchSize; idEnd = (x+1)*batchSize;
		rgb = np.asarray(dfile["rgb"]["rgb_data"][:,:,:,idStart:idEnd])
		depth = np.asarray(dfile["depth"]["depth_data"][:,:,idStart:idEnd])

		rgb = np.moveaxis(rgb, [0,1,2,3], [3,0,1,2])
		depth = np.moveaxis(depth, [0,1,2], [2,0,1])

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
	safe_mkdir(summaryLoc)
	safe_mkdir(weightsLoc)
	
def main():
	# Make network
	print("Loading network configuration")
	network = DEN(summaryLoc, weightsLoc, confFileName = configFile, training = True)
	# Prepare training data
	print("Preparing data")
	dataGenerator = prepare_data(dataFn, network.config["batchSize"])
	# Train network
	network.train(dataGenerator)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()