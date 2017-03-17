# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 13:59:42
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-17 17:19:33

import tensorflow as tf
import numpy as np

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
	transformed = tf.einsum('aij,i->aij', transformed, (image_shape - 1))
	return transformed

# TODO: implement this
def warp_using_coords(inGray, omega_dn):
	"""
		Warps image using the coordinates found in omega_dn
	"""
	return None

def warp_graph(depthFlat, inGray, poseM, shape):
	"""
		poseM: shape = [batch x 4 x 4]
	"""
	imageShapeTensor = tf.constant(shape) # TODO: See if necessary
	pixelAmount = shape[0] * shape[1]
	flat = lambda y: lambda x: shape[1]*y + x #Translates y,x to single index
	expand = lambda z: (z//shape[1],z % shape[1]) # Reverses previous operation

	# Generate pixel points
	pp = np.stack(expand(np.arange(pixelAmount)), axis=1)
	
	# Normalize pixel points
	ppNorm = normalize_image_points(pp, image.shape)
	# Convert to tensorflow
	ppNormTensor = tf.constant(ppNorm)
	# FIXME: Have a think to see if this still works in batches
	# TODO: See if it's possible to convert to einsum
	positionV = tf.stack([ppNorm[:,1]*inGraph, ppNorm[:,0]*inGraph, inGraph, tf.ones(pixelAmount)], axis = 0)
	# Use tf.einsum for batch matmul
	projectedPoints = tf.einsum('aij,ajk->aik', poseM, positionV)
	# FIXME: Won't work in batches
	omega = np.stack([projectedPoints[1]/projectedPoints[2], projectedPoints[0]/projectedPoints[2]], axis = 0)

	# keep track of what will be out of bounds in the image
	# Temporary variable of shape [batch x 2 x pixels]. Second axis denotes x and y positions, if either are out of bounds the pixel is as well
	oobPixels2D = tf.logical_or(omega > 1, omega < -1) 
	oobPixels = tf.logical_or(oobPixels2D[:,0,:], oobPixels2D[:,1,:])

	omega_dn = tf.round(denormalize_image_points(omega, imageShapeTensor))

	warped = warp_using_coords(inGray, omega_dn)
