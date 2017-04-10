# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-04 16:54:28
import numpy as np
import argparse
from nets.Depth_Estim_Net import Depth_Estim_Net as DEN
from util.util import loadmat, safe_mkdir
import data.NYU_Data as NYU_Data
import h5py

# TODO: Make argument parser

configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

depthFile = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_train2.hdf5"
rootData = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/"

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/fixedRelu_L2/"
summaryLoc = "%ssummary" % rootFolder
weightsLoc = "%scheckpoint/" % rootFolder

def prepare_data(depthFile, rootDataFolder, config):
	"""
	"""
	# Prepare generator with training split
	print("Preparing data generator")
	return NYU_Data.NYU_Data(rootDataFolder, depthFile, config["batchSize"])

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
	network.train(dataGenerator, loadChkpt = False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()
