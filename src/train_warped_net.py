# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-04-04 16:47:50
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-05 10:23:41

from nets.Slam_Loss_Net import Slam_Loss_Net
from data.ESAT_Warp_Data import ESAT_Warp_Data
from util.util import safe_mkdir
import tensorflow as tf

xmlPath = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/data/office2-4.xml"

configFile = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

netBasePath = "/esat/citrine/troussel/IROS/depth_estim/init_warp"
summaryLoc = "%s/summary" % netBasePath
weightsLoc = "%s/checkpoint/" % netBasePath

def prepare_dir():
	safe_mkdir(netBasePath)
	safe_mkdir(summaryLoc)
	safe_mkdir(weightsLoc)

def main():
	# Define net
	print("Loading network configuration")
	net = Slam_Loss_Net(weightsLoc, summaryLocation = summaryLoc, confFileName = configFile)
	# Data grabber object
	print("Preparing data grabber")
	data = ESAT_Warp_Data(xmlPath, net.config["batchSize"], (net.config["H"],net.config["W"]))
	prepare_dir()

	print("Training starting")
	for x in range(5):
		with tf.Graph().as_default():
			net.train(data, loadChkpt = True)


if __name__ == '__main__':
	main()