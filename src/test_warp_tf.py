# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-16 16:09:11

import numpy as np
import util.SEDWarp as SEDWarp
import matplotlib.pyplot as plt

def warpNormal(frame, depth, poseM):
	return SEDWarp.warp_image_opt(frame,depth,poseM)

def warpTF(frame, depth, poseM):
	# Make placeholder graph
	
	# Build warping graph

	# Properly reshape frame and depth

	# Start tensorflow session

		# Run graph

	return None

def main():
	# Load single image-pair
	(poseM, keyframe, frame, idName) = SEDWarp.decode_xml(fnXML, 5, bpath)
	depth = util.get_depth(idName, bpathDepth)
	# Warp image using normal implementation
	warpedNormal = warpNormal(frame, depth, poseM)
	# Warp image using tensorflow implementation
	warpedTF = warpTF(frame, depth, poseM)
	# Show results for both
	plt.subplt(131)
	plt.imshow(warpedNormal, cmap = 'gray')
	plt.title("Normal warping")
	plt.subplt(132)
	plt.imshow(warpedTF, cmap = 'gray')
	plt.title("TensorFlow warping")
	plt.subplt(133)
	plt.imshow(warpedTF - warpedNormal, cmap = 'gray')
	plt.title("Difference")
	plt.show()

if __name__ == '__main__':
	main()