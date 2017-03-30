# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-30 17:17:08
import tensorflow as tf
import numpy as np
import util.SEDWarp as SEDWarp
import util.SEDWarp_nonTF as SEDWarp2
import matplotlib.pyplot as plt
import util.util as util
from scipy.misc import imresize


fnXML = "/users/visics/troussel/tmp/test_SIM3.xml"
bpath = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/RGB"
bpathDepth = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/depth"


def warpTF(frame, depth, poseM):
	# Make placeholder graph
	inputGraph = tf.placeholder(tf.float32, shape = [2, 480, 640], name="RGB_in")
	depthGraph = tf.placeholder(tf.float32, shape = [2, 480, 640], name="depth_in")
	poseMGraph = tf.placeholder(tf.float32, shape = [2,4,4], name="poseM")
	# Build warping graph
	warpGraph = SEDWarp.warp_graph(depthGraph, inputGraph, poseMGraph)
	gradGraph = tf.gradients(warpGraph, depthGraph)

	# Start tensorflow session
	with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
		# Run graph
		result, grad = sess.run((warpGraph, gradGraph), feed_dict={inputGraph:frame, depthGraph: depth, poseMGraph: poseM})

	return result, grad

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
	# Warp image using tensorflow implementation
	frames = np.stack([frame1, frame2], axis=0)
	depths = np.stack([depth1, depth2], axis=0)
	poses = np.stack([poseM1, poseM2], axis=0)
	warpedTF, grads = warpTF(frames, depths, poses)
	# print warpedTF
	# Show results for both
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()