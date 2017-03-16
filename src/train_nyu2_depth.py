# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-14 11:14:35
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN
from util.util import loadmat, safe_mkdir
from math import floor
from glob import glob
from random import shuffle
import os.path
import scipy.misc

# TODO: Make argument parser

trainSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/train"
validateSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/validate"
testSetManifest = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/NYU2Split/test"

configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

labelFolder = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/labels_filled_proc/"
rootData = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/"

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/NYU/"
summaryLoc = "%ssummary" % rootFolder
weightsLoc = "%scheckpoint" % rootFolder

def load_batch(batch, rgbFiles, labelFolder, config):
	rgb = np.ones((config["batchSize"],config["H"],config["W"], 3))
	depth = np.ones((config["batchSize"],config["HOut"],config["WOut"]))

	i = 0
	lastPath = None
	for entry in batch:
		name, matPath = entry.split(',')
		# Get rgb file
		num = int(os.path.split(matPath)[-1].split("_")[1])
		match = "%s/imgs_%d/%s.jpeg" % (rgbFiles, num, name)
		rgb[i,:,:,:] = scipy.misc.imread(match, mode = "RGB")
		# Get GT depth info
		if not lastPath == matPath:
			# Load new file
			mat = loadmat(matPath)
			lastPath = matPath

		depth[i,:,:] = mat["labels_processed"][:,:,mat["labels_name"].tolist().index(name)]
		i += 1

	return rgb, depth

def train_data_generator(manifest, rootDataFolder, labelFolder, config):

	# Split manifest in batches
	batchAm = int(floor(len(manifest)/config["batchSize"]))
	# Loop over batches
	for x in xrange(batchAm):
		# Get Batch
		batch = manifest[x*config["batchSize"]:(x+1)*config["batchSize"]]
		# Sort batch
		batch = sorted(batch, key= lambda batch: batch.split(',')[1])
		# Load data
		print("Loading batch")
		rgb, gtDepth = load_batch(batch, rootDataFolder, labelFolder, config) 
		print("Batch loaded")
		# yield data
		yield rgb, gtDepth

def all_labels(labelFolder):
	allFiles = glob(labelFolder + "/labels_*.mat")
	filteredFiles = [x for x in allFiles if not "_esat" in x]

	allLabels = []
	for f in filteredFiles:
		mat = loadmat(f)
		for label in mat["labels_name"]:
			allLabels.append(label + ",%s" % f)

	return allLabels

def write_manifest(fn, labels):
	with open(fn, 'w') as f:
		fstring = "\n".join(labels)
		f.write(fstring)

def prepare_data(labelFolder, rootDataFolder, split, config, trainSetManifest, validateSetManifest=None, testSetManifest=None):
	"""
	"""
	assert split[0] + split[1] + split[2] == 1
	# Check for manifest
	if not os.path.exists(trainSetManifest):
		# Load all labels
		print("Fetching labels")
		labels = all_labels(labelFolder)
		# Make training/test/validate split
		print("Shuffling labels randomly")
		shuffle(labels)
		labelAm = len(labels)
		trainSet = labels[0:int(floor(labelAm*split[0]))]
		validateSet = labels[int(floor(labelAm*split[0])) : int(floor(labelAm*(split[0] + split[1])))]
		testSet = labels[int(floor(labelAm*(split[0] + split[1]))):]
		# Write manifest files
		print("Writing manifests")
		write_manifest(trainSetManifest, trainSet)
		write_manifest(validateSetManifest, validateSet)
		write_manifest(testSetManifest, testSet)
	else:
		# Load trainSet
		print("Loading manifest from file")
		with open(trainSetManifest, 'r') as f:
			fstring = f.read()
			trainSet = fstring.split("\n")

	# Prepare generator with training split
	print("Preparing data generator")
	return train_data_generator(trainSet, rootDataFolder, labelFolder, config)

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
	dataGenerator = prepare_data(labelFolder, rootData, (0.7,0.15,0.15), network.config, trainSetManifest, validateSetManifest, testSetManifest)
	# Train network
	print("Starting training")
	network.train(dataGenerator, loadChkpt = True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()
