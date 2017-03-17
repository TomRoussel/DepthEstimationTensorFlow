# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-09 16:04:29
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-16 16:31:40
from Depth_Estim_Net import Depth_Estim_Net as DEN
import numpy as np
from util.ESAT_Data import ESAT_Data
import matplotlib.pyplot as plt
import h5py

config = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"
dataFn = "/esat/citrine/tmp/troussel/IROS/kinect/StijnKinectDataNP/1.hdf5" 

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/NYU_BN_scope/" 
weightsLoc = "%scheckpoint/" % rootFolder

threshold = 1.25

def main():
	# Load Network
	net = DEN(weightsLoc, summaryLocation = "/users/visics/troussel/tmp/sum/", confFileName = config, training = False)
	net.config["batchSize"] =  128

	# Construct data loader
	data = ESAT_Data(dataFn, net.config)

	rmseList = []
	validCount = 0
	totalCount = 0
	eps = 1e-9
	for index, (rgb, gtDepth) in enumerate(data):
		# Frop
		depth = net.fprop(rgb)
		# Get loss
		diff = depth - np.reshape(gtDepth, (net.config["batchSize"], net.config["HOut"]*net.config["WOut"]))
		rmse = np.sum(np.power(diff,2))/ (diff.shape[0] * diff.shape[1])

		# Calculate accuracy as defined in cnn based single image depth paper
		delta = np.maximum(depth.flatten()/(gtDepth.flatten() + eps), gtDepth.flatten()/(depth.flatten() + eps))
		# Threshold
		validCount += np.sum(delta < threshold)
		totalCount += delta.shape[0]

		rmseList.append(rmse)
		print("Processed batch %d of %d" % (index, len(data)))

	fullRMSE = np.mean(rmseList)
	print("fullRMSE is %f\nAccuracy is %f%%" % (fullRMSE, validCount/float(totalCount) * 100))

if __name__ == '__main__':
	main()
