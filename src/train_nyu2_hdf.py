# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-14 10:46:25
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN
from util.util import loadmat
import util.NYU_Data as NYU_Data
import h5py

# TODO: Make argument parser

trainSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/train"
validateSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/validate"
testSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/test"

configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

depthFile = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_train2.hdf5"
rootData = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/"

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/NYU_BN_scope/"
summaryLoc = "%ssummary" % rootFolder
weightsLoc = "%scheckpoint/" % rootFolder

def prepare_data(depthFile, rootDataFolder, config):
	"""
	"""
	# assert split[0] + split[1] + split[2] == 1
	hdfF = h5py.File(depthFile, 'r')

	# Prepare generator with training split
	print("Preparing data generator")
	return NYU_Data.train_data_generator(rootDataFolder, hdfF, config)


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
	network = DEN(weightsLoc, summaryLoc, confFileName = configFile, training = True)
	# Prepare training data
	print("Preparing data")
	dataGenerator = prepare_data(depthFile, rootData, network.config)
	# Train network
	print("Starting training")
	network.train(dataGenerator, loadChkpt = True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()
