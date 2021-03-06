# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-04-03 15:49:47
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-11 17:13:33

import numpy as np
from math import floor
from util.SEDWarp import decode_frame_info
import xml.etree.ElementTree as ET
from scipy.misc import imresize

# NOTE: Can most likely be sped up by converting everything to an HDF file
class ESAT_Warp_Data(object):
	"""
		This class handles loading the ESAT sequence 
		dataset recorded by the kinect2
	"""
	def __init__(self, fnXML, batchSize, shape, invertPose = False):
		"""
			@fnXML      : filename of xml file containing data paths and pose matrix
			@batchSize  : desired size of each batch
			@shape	    : desired output shape of the keyframe and frame
			@invertPose : if set to True, keyframe and frame will be swapped and posematrix will be inverted
		"""
		self.fnXML = fnXML
		self.batchSize = batchSize
		# Get amount of batches
		self.batchAm = floor(len(ET.parse(fnXML).getroot())/batchSize)
		self.shape = shape
		self.invertPose = invertPose
		print("File contains %d batches using a batchsize of %d" % (self.batchAm, batchSize))		


	def __getitem__(self, key):
		if key >= self.batchAm:
			raise IndexError
		else:
			# Get data from frame
			startI = key * self.batchSize
			endI = (key + 1) * self.batchSize
			keyframes = -np.ones((self.batchSize, self.shape[0], self.shape[1], 3))
			frames = -np.ones((self.batchSize, self.shape[0], self.shape[1], 3))
			poseMs = np.ones((self.batchSize, 4, 4))

			frameInfoList = ET.parse(self.fnXML).getroot()[startI:endI]
			for index, frame in enumerate(frameInfoList):
				(poseMFrame, keyframe, frame, _) = decode_frame_info(frame)
				poseMs[index, :,:] = poseMFrame
				keyframes[index,:,:,:] = imresize(keyframe, self.shape)
				frames[index,:,:,:] = imresize(frame, self.shape)

			if not self.invertPose:
				return keyframes, poseMs, frames
			else:
				poseMs = np.linalg.inv(poseMs)
				return frames, poseMs, keyframes

	def __len__(self):
		return self.batchAm