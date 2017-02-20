# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-02-07 17:22:39
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-02-07 17:53:55

import numpy as np
import h5py
import scipy.io as sio


path = "/esat/emerald/pchakrav/StijnData/ESATKinectImagesDataset/skeletonized_dataset/"
outPath = "/esat/citrine/tmp/troussel/IROS/kinect/StijnKinectDataNP/1.hdf5"

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

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
		fdata = loadmat(path + filename)
		depth[:,:,startI:endI] = fdata["data"]["depth_map"]
		rgb[:,:,:,startI:endI] = fdata["data"]["rgb"]

if __name__ == '__main__':
	main()