# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 17:22:39
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-02 10:16:09

import numpy as np
import h5py
import sys; sys.path.append("../util")
import util
from glob import glob
import os.path

outPath = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU.hdf5"
labelFolder = "/esat/emerald/pchakrav/StijnData/NYUv2/processed/labels_filled_proc/"

def all_labels(labelFolder):
	allFiles = glob(labelFolder + "/labels_*.mat")
	filteredFiles = [x for x in allFiles if not "_esat" in x]

	allLabels = []
	for f in filteredFiles:
		mat = util.loadmat(f)
		for label in mat["labels_name"]:
			allLabels.append(label + ",%s" % f)

	return allLabels

def main():
	print("Fetching labels")
	labels = all_labels(labelFolder)

	f = h5py.File(outPath, 'w')
	depth = f.create_group("depth").create_dataset("depth_data",(55, 74,len(labels)), compression="gzip", dtype=np.float64)
	dt = h5py.special_dtype(vlen=bytes)
	depth_label = f["depth"].create_dataset("depth_labels", [len(labels)], dtype=dt)
	depth_mat_source = f["depth"].create_dataset("depth_folder_id", [len(labels)], dtype=np.uint16)
	
	allFiles = glob(labelFolder + "/labels_*.mat")
	filteredFiles = [x for x in allFiles if not "_esat" in x]

	startI = 0

	fileNo = 1

	# Loop over all data
	for x in filteredFiles:
		print("Processing file %d/%d" % (fileNo, len(filteredFiles)))
		matFileNo = int(os.path.split(x)[-1].split("_")[1])
		fdata = util.loadmat(x)
		endI = startI + fdata["labels_name"].shape[0]
		depth[:,:,startI:endI] = fdata["labels_processed"]
		depth_label[startI:endI] = fdata["labels_name"]
		depth_mat_source[startI:endI] = matFileNo

		# Update startI
		startI = endI

		fileNo += 1

if __name__ == '__main__':
	main()