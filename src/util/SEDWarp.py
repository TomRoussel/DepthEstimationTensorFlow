# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 13:59:42
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-03 12:33:04

import tensorflow as tf
import numpy as np
from ops import ZeroOutOps
import xml.etree.ElementTree as ET
from scipy.misc import imread

zero_out3 = ZeroOutOps.zero_out3

# TODO: properly deal with invalid pixels

def _decode_frame_info(frameInfo, bpath):
	# Parse the pose matrix
	poseMText = frameInfo.find("pose_matrix").text
	poseMList = [float(x) for x in poseMText.split(',')]
	poseM = np.reshape(np.asarray(poseMList), (4,4))

	# Get the keyframe image
	kfText = frameInfo.find("kf_path").text
	kfloc = "%s/%s.png" % (bpath, kfText)
	kf = imread(kfloc)

	# Get the compared image
	fText = frameInfo.find("f_path").text
	floc = "%s/%s.png" % (bpath, fText)
	f = imread(floc)

	return (poseM, kf, f, kfText)

def decode_xml(fnXML, index, bpath):
	# Load xml file
	tree = ET.parse(fnXML)
	frameInfo = tree.getroot()[index]
	
	return _decode_frame_info(frameInfo, bpath)

def image_gradient(image):
	return np.gradient(image, axis = (1,2))

@tf.RegisterGradient("ZeroOut3")
def test_gradient_register(operation, grad):
	image, omega, oobPixels = operation.inputs
	batchSize = int(image.shape[0])
	shape = [int(x) for x in image.shape][1:]
	pixelAmount = shape[0] * shape[1]

	omegaRnd = tf.round(omega)
	# Flatten omega
	b = tf.constant([int(shape[1]),1], dtype=tf.float32)
	omegaFlat = tf.einsum('aij,i->aj', omegaRnd, b) # Shape: [batches x pixels]

	omegaFlat = tf.reshape(omegaFlat, (batchSize,-1))
	omegaFlat = tf.round(omegaFlat)
	omegaFlat = omegaFlat * tf.cast(tf.logical_not(oobPixels), dtype = tf.float32)
	batchNo = tf.tile(tf.reshape(tf.linspace(0.0,float(batchSize-1), batchSize), (-1,1)), (1,pixelAmount))
	
	indexes = tf.stack([batchNo,omegaFlat], axis = 2)
	indexes = tf.cast(indexes, tf.int32)
	# indexes = tf.Print(indexes, [indexes], message="indexes", summarize = 1e6)

	# Get gradient image
	gradientImage = tf.stack(tf.py_func(image_gradient, [image], [tf.float32,tf.float32]), axis=1) # [batches, dimensions, H, W]
	gradientImageFlat = tf.reshape(gradientImage, (batchSize, 2, pixelAmount))
	# Evaluate with omega
	fx = tf.gather_nd(gradientImageFlat[:,0,:], indexes)
	fy = tf.gather_nd(gradientImageFlat[:,1,:], indexes)
	gradFlat = tf.reshape(grad, (batchSize, pixelAmount))

	outGrad = tf.stack([fx*gradFlat, fy*gradFlat], axis = 1)

	return None, outGrad, None

def normalize_image_points(coords, image_shape):
	"""
		For now, assuming calibration is already done
	"""
	coords = np.asarray(coords, dtype=np.float32)
	normalized = np.divide(coords, np.reshape(np.asarray(image_shape), (2,1))-1)
	normalized = (normalized*2)-1
	return normalized

def denormalize_image_points(coords, image_shape):
	"""
		returns the absolute coordinates (corresponds to matrix indexes)
	"""
	transformed = (coords + 1) / 2.0
	transformed = tf.einsum('aij,i->aij', transformed, (image_shape - 1))
	return transformed

def _indexing_op_(inGray, omega):
	"""
		Performs the actual warping
		@inGray: input grayscale image shape [batchSize, pixels]
		@omega: Warping coordinates, shape [batchSize, pixels]
		@shape: The output shape
	"""
	batchSize = int(omega.shape[0])
	omega = tf.reshape(omega, (batchSize,-1))
	omega = tf.round(omega)
	batchNo = tf.tile(tf.reshape(tf.linspace(0.0,float(batchSize-1), batchSize), (-1,1)), [1,int(inGray.shape[2])*int(inGray.shape[1])])
	indexes = tf.stack([batchNo,omega], axis = 2)
	inGrayFlat = tf.reshape(inGray, (batchSize, int(inGray.shape[2])*int(inGray.shape[1])))
	warpedFlat = tf.gather_nd(inGrayFlat, tf.cast(indexes, tf.int32)) # NOTE: no gradients through indices
	warped = tf.reshape(warpedFlat, (batchSize, int(inGray.shape[1]), int(inGray.shape[2])))
	return warped

def warp_using_coords(inGray, omegaDN, oobPixels):
	"""
		Warps image using the coordinates found in omegaDN
		@inGray: input grayscale image shape [batchSize, pixels]
		@omega: Warping coordinates, shape [batchSize, 2, pixels]
		@oobPixels: out of bounds pixels, entries set to true will be 0 in the output image, shape [batchSize, pixels]
		@shape: The output shape
	"""
	omegaDNRnd = tf.round(omegaDN)
	batchSize = int(omegaDN.shape[0])
	# Flatten omegaDN
	b = tf.constant([int(inGray.shape[2]),1], dtype=tf.float32)
	omegaFlat = tf.einsum('aij,i->aj', omegaDNRnd, b) # Shape: [batches x pixels]

	# Only valid pixels
	omegaFlat_VO = omegaFlat * tf.cast(tf.logical_not(oobPixels), dtype = tf.float32)

	warped = _indexing_op_(inGray, omegaFlat_VO)
	return warped

def warp_graph(depth, inGray, poseM):
	"""
		Constructs a graph that warps an image for a given depth and pose matrix
		@depth: Depth image, flattened. Shape [batchSize, pixels]
		@inGray: input grayscale image, will be warped. Shape [batchSize, pixels]
		@poseM: Posematrix representing the movement between two frames. Shape [batchSize, 4, 4]
		@shape: output shape
	"""
	batchSize = [int(x) for x in inGray.shape][0]
	shape = [int(x) for x in inGray.shape][1:]
	pixelAmount = shape[0] * shape[1]
	expand = lambda z: (z//shape[1],z % shape[1]) # Reverses previous operation

	assert len(shape) == 2, "Unexpected shape: inGray"

	depthFlat = tf.reshape(depth, (batchSize, pixelAmount))
	# Add small value to prevent division by zero later on
	depthFlat += 1e-9

	# Generate pixel points
	pp = np.stack(expand(np.arange(pixelAmount)), axis=0)
	
	# Normalize pixel points
	ppNorm = normalize_image_points(pp, shape)
	# Convert to tensorflow
	ppNormTensor = tf.concat([tf.constant(np.tile(ppNorm, (batchSize, 1,1)), dtype=tf.float32), tf.ones((batchSize, 1, pixelAmount))], axis = 1)

	# Multiply by depth values
	positionV = tf.einsum('abk,ak->abk', ppNormTensor, depthFlat)
	positionV = tf.stack([positionV[:,1,:],positionV[:,0,:],positionV[:,2,:]], axis = 1)
	# Add row of ones
	positionV = tf.concat([positionV, tf.ones((batchSize, 1, pixelAmount))], axis = 1)

	# Use tf.einsum for batch matmul
	projectedPoints = tf.einsum('aij,ajk->aik', poseM, positionV)

	omega = tf.stack([projectedPoints[:,1,:]/projectedPoints[:,2,:], projectedPoints[:,0,:]/projectedPoints[:,2,:]], axis = 1)

	# Keep track of what will be out of bounds in the image
	# Temporary variable of shape [batch x 2 x pixels]. Second axis denotes x and y positions, if either are out of bounds the entire pixel is
	oobPixels2D = tf.logical_or(omega >= 1, omega <= -1) 
	oobPixels = tf.logical_or(oobPixels2D[:,0,:], oobPixels2D[:,1,:])

	omega_dn = denormalize_image_points(omega, tf.constant(shape, dtype=tf.float32))
	# omega_dn = tf.Print(omega_dn, [ppNormTensor, positionV, projectedPoints, omega, oobPixels], summarize = 1e6)
	warped = warp_using_coords(inGray, omega_dn, oobPixels)
	# Override gradient
	B = zero_out3(warped, omega_dn, oobPixels)
	warped = B + tf.stop_gradient(warped - B)
	return warped
