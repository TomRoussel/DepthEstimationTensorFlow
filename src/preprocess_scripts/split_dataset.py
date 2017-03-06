import numpy as np
import h5py
import os.path
from random import shuffle
from math import floor

inPath = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_lzif.hdf5"
trainPath = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_train2.hdf5"
validatePath = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_validate2.hdf5"
testPath = "/esat/citrine/tmp/troussel/IROS/kinect/NYU_data/NYU_test2.hdf5"

split = (0.7,0.15,0.15)

def write_new_file(inF, outF, indexes, maxLoadSize):
	depth = outF.create_group("depth").create_dataset("depth_data",(55, 74,len(indexes)), compression="lzf", dtype=np.float64)
	dt = h5py.special_dtype(vlen=bytes)
	depth_label = outF["depth"].create_dataset("depth_labels", [len(indexes)], dtype=dt)
	depth_mat_source = outF["depth"].create_dataset("depth_folder_id", [len(indexes)], dtype=np.uint16)

	depth_in = inF["depth"]["depth_data"]
	depth_label_in = inF["depth"]["depth_labels"]
	depth_mat_source_in = inF["depth"]["depth_folder_id"]

	i = 0
	startI = 0
	while startI < len(indexes):
		endI = startI + maxLoadSize
		endI = endI if endI < len(indexes) else len(indexes)

		inputI = indexes[startI:endI]
		inputI = sorted(inputI)
		depth[:,:,startI:endI] = depth_in[:,:,inputI]
		depth_label[startI:endI] = depth_label_in[inputI]
		depth_mat_source[startI:endI] = depth_mat_source_in[inputI]

		startI = endI
		if i % 10 == 0:
			print("Copying to new file: %f%% done" % (float(startI) / len(indexes) * 100.0))
		i += 1
	

def main():

	f = h5py.File(inPath, 'r')
	print("Generating splits")
	imAmount = f["depth"]["depth_labels"].shape[0]
	shuffledIndexes = range(imAmount); shuffle(shuffledIndexes)
	trainIndexes = shuffledIndexes[0:int(floor(split[0] * imAmount))]
	testIndexes = shuffledIndexes[int(floor(split[0] * imAmount)):int(floor((split[0]+split[1]) * imAmount))]
	validateIndexes = shuffledIndexes[int(floor((split[0]+split[1]) * imAmount)):-1]
	
	fTrain = h5py.File(trainPath, 'w')
	fTest = h5py.File(testPath, 'w')
	fValidate = h5py.File(validatePath, 'w')

	print("Writing training file")
	write_new_file(f, fTrain, trainIndexes, 20)
	print("Writing testing file")
	write_new_file(f, fTest, testIndexes, 2000)
	print("Writing validation file")
	write_new_file(f, fValidate, validateIndexes, 2000)

	

if __name__ == '__main__':
	main()