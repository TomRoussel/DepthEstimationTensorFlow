# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 13:59:42
# @Last Modified by:   Tom
# @Last Modified time: 2017-03-24 11:21:35

import tensorflow as tf
import numpy as np

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

# TODO: Clean this up. There's no reason for this to have so many parameters
# TODO: Implement gradient for this function
def warp_using_coords(inGray, omega_dn, shape, batchSize, oobPixels):
	"""
		Warps image using the coordinates found in omega_dn
	"""
	# Flatten omega_dn
	b = tf.constant([shape[1],1], dtype=tf.float32)
	omegaFlat = tf.einsum('aij,i->aj', omega_dn, b) # Shape: [batches x pixels]

	# Only valid pixels
	omegaFlat_VO = omegaFlat * tf.cast(tf.logical_not(oobPixels), dtype = tf.float32)

	omegaFlat_VO = tf.reshape(omegaFlat_VO, (batchSize,-1))

	batchNo = tf.tile(tf.reshape(tf.linspace(0.0,float(batchSize-1), batchSize), (-1,1)), [1,shape[1]*shape[0]])
	indexes = tf.stack([batchNo,omegaFlat_VO], axis = 2)

	warped = tf.gather_nd(inGray, tf.cast(indexes, tf.int32)) # NOTE: no gradients through indices
	return warped

# FIXME: Test extent of automatic gradient calculation
def warp_graph(depthFlat, inGray, poseM, shape, batchSize):
	"""
		poseM: shape = [batch x 4 x 4]
	"""
	pixelAmount = shape[0] * shape[1]
	expand = lambda z: (z//shape[1],z % shape[1]) # Reverses previous operation

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

	omega_dn = tf.round(denormalize_image_points(omega, tf.constant(shape, dtype=tf.float32))) # NOTE: no gradients through round
	# omega_dn = tf.Print(omega_dn, [ppNormTensor, positionV, projectedPoints, omega, oobPixels], summarize = 1e6)
	warpedFlat = warp_using_coords(inGray, omega_dn, shape, batchSize, oobPixels)
	return tf.reshape(warpedFlat, (batchSize, shape[0], shape[1]))
