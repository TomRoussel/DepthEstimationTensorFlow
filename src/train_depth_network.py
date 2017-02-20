# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 16:14:25
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-02-09 10:19:12
import numpy as np
import argparse
from Depth_Estim_Net import Depth_Estim_Net as DEN

# TODO: Data prepping

def prepare_data():
	

	
def main():
	# Prepare training data
	print("Preparing data")
	rgb, depth = prepare_data()
	# Make network
	print("Loading network configuration")
	network = DEN(confFileName = configFile, training = True)
	# Train network
	network.train_dataset(rgb, depth)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	main()