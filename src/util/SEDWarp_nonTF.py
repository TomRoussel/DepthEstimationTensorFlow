# -*- coding: utf-8 -*-
# @Author: troussel
# @Date:   2017-02-28 12:13:28
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-16 17:11:37
# Library for warping images with pose and depth information

import numpy as np
import xml.etree.ElementTree as ET
from scipy.misc import imread
from scipy.optimize import minimize

def _decode_frame_info(frameInfo, bpath):
	# Parse the pose matrix
	poseMText = frameInfo.find("pose_matrix").text
	poseMList = [float(x) for x in poseMText.split(',')]
	poseM = np.reshape(np.asarray(poseMList), (4,4))

	# Get the keyframe image
	kfText = frameInfo.find("kf_path").text
	kfloc = "%s/%s.png" % (bpath, kfText)
	kf = imread(kfloc, flatten = True)

	# Get the compared image
	fText = frameInfo.find("f_path").text
	floc = "%s/%s.png" % (bpath, fText)
	f = imread(floc, flatten = True)

	return (poseM, kf, f, kfText)

def decode_xml(fnXML, index, bpath):
	# Load xml file
	tree = ET.parse(fnXML)
	frameInfo = tree.getroot()[index]
	
	return _decode_frame_info(frameInfo, bpath)

def xml_entries(fnXML, bpath):
	"""
		Generator function, iterates over all entries in the given XML files
	"""
	tree = ET.parse(fnXML)
	root = tree.getroot()
	
	for frame in root:
		yield _decode_frame_info(frame, bpath)


# TODO: Maybe finish converting warping to an OO method
		
def normalize_image_points(coords, image_shape):
	"""
		For now, assuming calibration is already done
	"""
	coords = np.asarray(coords, dtype=np.float32)
	normalized = np.divide(coords, np.asarray(image_shape)-1)
	normalized = (normalized*2)-1
	return normalized

def denormalize_image_points(coords, image_shape):
	"""
		returns the absolute coordinates (corresponds to matrix indexes)
	"""
	transformed = (coords + 1) / 2.0
	transformed = np.multiply(transformed, np.asarray(image_shape)[:,np.newaxis] - 1)
	return transformed

def warp_image_opt(image, depth, poseM, returnCorrMap = False, returnInvalidPixels = False):
	"""
		Warps an image based on the depth map and the lie group notation of the pose vector
		If the depth contains invalid data entries, a function can be passed through invalidDataFunc which should return a boolean matrix with False = valid pixel and True = invalid pixel
	"""
	assert image.shape == depth.shape
	# Initialize
	warpedImage = np.zeros(image.shape)
	height, width = image.shape
	pixelAmount = height * width
	eps = 1e-9
	# Flatten arrays
	imageFlat = np.reshape(image, pixelAmount)
	depthFlat = np.reshape(depth, pixelAmount)
	depthFlat = depthFlat + eps
	flat = lambda y: lambda x: image.shape[1]*y + x #Translates y,x to single index
	expand = lambda z: (z//image.shape[1],z % image.shape[1]) # Reverses previous operation

	# Generate pixel points
	pp = np.stack(expand(np.arange(pixelAmount)), axis=1)
	
	# Normalize pixel points
	ppNorm = normalize_image_points(pp, image.shape)
	# Construct position vector
	v = np.stack([ppNorm[:,1]*depthFlat, ppNorm[:,0]*depthFlat, depthFlat, np.ones(pixelAmount)], axis = 0)
	# Project
	projected = np.dot(poseM, v)
	# Get omega
	omega = np.stack([projected[1]/projected[2], projected[0]/projected[2]], axis = 0)
	invalidPixelsTmp = np.logical_or(omega > 1, omega < -1)
	invalidPixels = np.logical_or(invalidPixelsTmp[0], invalidPixelsTmp[1])
	# Denormalize points
	omega_dn = denormalize_image_points(omega, image.shape)
	omega_dn = np.round(omega_dn).astype(np.int32)
	omega_dn[:, invalidPixels] = 0

	corrMap = np.reshape(omega_dn.T, (width, height, 2), order='F')
	corrMap = np.swapaxes(corrMap, 1,0)
	# Return image
	warpedImageFlat = imageFlat[flat(omega_dn[0])(omega_dn[1])]
	warpedImage = np.reshape(warpedImageFlat, (height,width), order='C')

	invalidPixelsImage = np.reshape(invalidPixels, (height,width))
	invalidPixelsImage = np.logical_or(invalidPixelsImage, depth == 0)
	if returnCorrMap and returnInvalidPixels:
		return (warpedImage, corrMap, invalidPixelsImage)
	elif returnCorrMap:
		return (warpedImage, corrMap)
	elif returnInvalidPixels:
		return (warpedImage, invalidPixelsImage)
	else:
		return warpedImage

def warp_image_error_min(image, depth, poseM, keyframe, errorFunc = None,returnCorrMap = False, returnInvalidPixels = False):
	if errorFunc is None:
		# Default error function is unaltered RMSE
		errorFunc = lambda x1: lambda x2: np.sqrt(np.sum(np.power(x1-x2, 2)))/(x1.shape[0]*x1.shape[1])

	costFunction = lambda scale: errorFunc(keyframe, warp_image_opt(image, scale*depth, poseM))

	res = minimize(costFunction, 0.02, method='Nelder-Mead', options={'disp': True}, bounds = ((0,None)))

	print("Found scale %f" % res.x)

	return warp_image_opt(image, res.x*depth, poseM, returnCorrMap = returnCorrMap, returnInvalidPixels = returnInvalidPixels)