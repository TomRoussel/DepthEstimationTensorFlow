# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-04-03 15:49:47
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-04 15:51:19

import numpy as np
from math import floor
from util.SEDWarp import decode_frame_info
import xml.etree.ElementTree as ET
from scipy.misc import imresize

class ESAT_Warp_Data(object):
	"""
		This class handles loading the ESAT sequence 
		dataset recorded by the kinect2
	"""
	def __init__(self, fnXML, batchSize, shape):
		self.fnXML = fnXML
		self.batchSize = batchSize
		# Get amount of batches
		self.batchAm = len(ET.parse(fnXML).getroot())
		self.shape = shape

	def __getitem__(self, key):
		if key >= self.batchAm:
			raise IndexError
		else:
			# Get data from frame
			startI = key * self.batchSize
			endI = (key + 1) * self.batchSize
			keyframes = -np.ones((self.batchSize, shape[0], shape[1], 3))
			frames = -np.ones((self.batchSize, shape[0], shape[1], 3))
			poseMs = np.ones((self.batchSize, 4, 4))

			frameInfoList = ET.parse(self.fnXML).getroot()[startI:endI]
			for index, frame in enumerate(frameInfoList):
				(poseMFrame, keyframe, frame, _) = decode_frame_info(frame)
				poseMs[index, :,:] = poseMFrame
				keyframes[index,:,:,:] = imresize(keyframe, shape)
				frames[index,:,:,:] = imresize(frame, shape)

			return keyframes, poseMs, frames

	def __len__(self):
		return self.batchAm