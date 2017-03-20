# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-20 15:14:00

import numpy as np
import util.SEDWarp as SEDWarp
import util.SEDWarp_nonTF as SEDWarp2
import matplotlib.pyplot as plt

def warpNormal(frame, depth, poseM):
	return SEDWarp2.warp_image_opt(frame,depth,poseM)

def warpTF(frame, depth, poseM):
	# Make placeholder graph
	inputGraph = tf.placeholder(tf.float32, shape = [1, 240, 320], name="RGB_in")
	depthGraph = tf.placeholder(tf.float32, shape = [1, 240, 320], name="depth_in")
	poseMGraph = tf.placeholder(tf.float32, shape = [1,4,4], name="poseM")
	# Properly reshape frame and depth
	greyFlat = tf.reshape(inputGraph, (1,240*320))
	depthFlat = tf.reshape(depthGraph, (1,240*320))
	# Build warping graph
	warpGraph = SEDWarp.warp_graph(depthFlat, greyFlat, poseM, (240,320), 1)

	# Start tensorflow session
	with tf.Session as sess:
		# Run graph
		result = sess.run(warpGraph, feed_dict={inputGraph:frame, depthGraph: depth, poseMGraph: poseM})

	return result

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