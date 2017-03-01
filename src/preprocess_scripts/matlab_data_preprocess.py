# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 17:22:39
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-01 16:47:57

import numpy as np
import h5py
import util.util


path = "/esat/emerald/pchakrav/StijnData/ESATKinectImagesDataset/skeletonized_dataset/"
outPath = "/esat/citrine/tmp/troussel/IROS/kinect/StijnKinectDataNP/1.hdf5"

def main():
	f = h5py.File(outPath, 'w')
	depth = f.create_group("depth").create_dataset("depth_data",(55, 74,22000), compression="gzip")
	rgb = f.create_group("rgb").create_dataset("rgb_data",  (240, 320, 3,22000), compression="gzip", dtype='uint8')

	perFile = 2000
	# Loop over all data
	for x in xrange(22):
		print("file %d/22" % (x+1))
		startI = x*perFile
		endI = (x+1)*perFile
		filename = "esat_%d_%d.mat" % (startI+1, endI)
		fdata = util.loadmat(path + filename)
		depth[:,:,startI:endI] = fdata["data"]["depth_map"]
		rgb[:,:,:,startI:endI] = fdata["data"]["rgb"]

if __name__ == '__main__':
	main()