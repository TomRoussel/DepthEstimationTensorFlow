# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-22 16:04:08
import tensorflow as tf
import numpy as np
import util.SEDWarp as SEDWarp
import util.SEDWarp_nonTF as SEDWarp2
import matplotlib.pyplot as plt
import util.util as util
from scipy.misc import imresize

from time import time

fnXML = "/users/visics/troussel/tmp/test_SIM3.xml"
bpath = "/esat/citrine/tmp/troussel/IROS/kinect/Kinect2/tmp_data/office4/RGB"
bpathDepth = "/esat/citrine/tmp/troussel/IROS/kinect/Kinect2/tmp_data/office4/depth"

# FIXME: Test batch warping

def warpNormal(frame, depth, poseM):
	t1 = time()
	warped = SEDWarp2.warp_image_opt(frame,depth,poseM)
	t2 = time()
	print("Numpy implementation took %f seconds" % (t2 - t1))
	return warped

def warpTF(frame, depth, poseM):
	# Make placeholder graph
	inputGraph = tf.placeholder(tf.float32, shape = [1, 240, 320], name="RGB_in")
	depthGraph = tf.placeholder(tf.float32, shape = [1, 240, 320], name="depth_in")
	poseMGraph = tf.placeholder(tf.float32, shape = [1,4,4], name="poseM")
	# Properly reshape frame and depth
	greyFlat = tf.reshape(inputGraph, (1,240*320))
	depthFlat = tf.reshape(depthGraph, (1,240*320))
	# Build warping graph
	warpGraph = SEDWarp.warp_graph(depthFlat, greyFlat, poseMGraph, (240,320), 1)

	# Start tensorflow session
	with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
		# Run graph
		t1 = time()
		result = sess.run(warpGraph, feed_dict={inputGraph:np.reshape(frame, (1,240,320)), depthGraph: np.reshape(depth, (1,240,320)), poseMGraph: np.reshape(poseM, (1,4,4))})
		t2 = time()

	print("TF implementation took %f seconds" % (t2 - t1))

	return result

def main():
	# Load single image-pair
	(poseM, keyframe, frame, idName) = SEDWarp2.decode_xml(fnXML, 5, bpath)
	depth = util.get_depth(idName, bpathDepth)
	frame = imresize(frame, (240,320))
	depth = imresize(depth, (240,320))
	# Warp image using normal implementation
	warpedNormal = warpNormal(frame, depth, poseM)
	# Warp image using tensorflow implementation
	warpedTF = warpTF(frame, depth, poseM)
	print warpedTF
	# Show results for both
	plt.subplot(131)
	plt.imshow(warpedNormal, cmap = 'gray')
	plt.title("Normal warping")
	plt.subplot(132)
	plt.imshow(warpedTF[0], cmap = 'gray')
	plt.title("TensorFlow warping")
	plt.subplot(133)
	plt.imshow(warpedTF[0] - warpedNormal, cmap = 'gray')
	plt.title("Difference")
	plt.show()

if __name__ == '__main__':
	main()