# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-29 16:31:59
import tensorflow as tf
import numpy as np
import util.SEDWarp as SEDWarp
import util.SEDWarp_nonTF as SEDWarp2
import matplotlib.pyplot as plt
import util.util as util
from scipy.misc import imresize

from time import time

fnXML = "/users/visics/troussel/tmp/test_SIM3.xml"
bpath = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/RGB"
bpathDepth = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/depth"

def warpNormal(frame, depth, poseM):
	warped = SEDWarp2.warp_image_opt(frame,depth,poseM)
	return warped

def warpTF(frame, depth, poseM):
	# Make placeholder graph
	inputGraph = tf.placeholder(tf.float32, shape = [2, 480, 640], name="RGB_in")
	depthGraph = tf.placeholder(tf.float32, shape = [2, 480, 640], name="depth_in")
	poseMGraph = tf.placeholder(tf.float32, shape = [2,4,4], name="poseM")
	# Properly reshape frame and depth
	# greyFlat = tf.reshape(inputGraph, (2,480*640))
	# depthFlat = tf.reshape(depthGraph, (2,480*640))
	# Build warping graph
	warpGraph = SEDWarp.warp_graph(depthGraph, inputGraph, poseMGraph)
	# print(tf.gradients(warpGraph, depthGraph))

	# Start tensorflow session
	with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
		# Run graph
		t1 = time()
		result = sess.run(warpGraph, feed_dict={inputGraph:frame, depthGraph: depth, poseMGraph: poseM})
		t2 = time()

	print("TF implementation took %f seconds" % (t2 - t1))

	return result

def main():
	# Load single image-pair
	(poseM1, keyframe1, frame1, idName1) = SEDWarp2.decode_xml(fnXML, 5, bpath)
	(poseM2, keyframe2, frame2, idName2) = SEDWarp2.decode_xml(fnXML, 75, bpath)
	depth1 = util.get_depth(idName1, bpathDepth)
	depth2 = util.get_depth(idName2, bpathDepth)

	frame1 = imresize(frame1, (480,640))
	depth1 = imresize(depth1, (480,640))
	frame2 = imresize(frame2, (480,640))
	depth2 = imresize(depth2, (480,640))
	# Warp image using normal implementation
	t1 = time()
	warpedNormal1 = warpNormal(frame1, depth1, poseM1)
	warpedNormal2 = warpNormal(frame2, depth2, poseM2)
	t2 = time()
	print("Numpy implementation took %f seconds" % (t2 - t1))
	# Warp image using tensorflow implementation
	frames = np.stack([frame1, frame2], axis=0)
	depths = np.stack([depth1, depth2], axis=0)
	poses = np.stack([poseM1, poseM2], axis=0)
	warpedTF = warpTF(frames, depths, poses)
	# print warpedTF
	# Show results for both
	plt.subplot(231)
	plt.imshow(warpedNormal1, cmap = 'gray')
	plt.title("Normal warping")
	plt.subplot(232)
	plt.imshow(warpedTF[0], cmap = 'gray')
	plt.title("TensorFlow warping")
	plt.subplot(233)
	plt.imshow(warpedTF[0] - warpedNormal1, cmap = 'gray')
	plt.title("Difference")
	plt.subplot(234)
	plt.imshow(warpedNormal2, cmap = 'gray')
	plt.title("Normal warping")
	plt.subplot(235)
	plt.imshow(warpedTF[1], cmap = 'gray')
	plt.title("TensorFlow warping")
	plt.subplot(236)
	plt.imshow(warpedTF[1] - warpedNormal2, cmap = 'gray')
	plt.title("Difference")
	plt.show()

if __name__ == '__main__':
	main()