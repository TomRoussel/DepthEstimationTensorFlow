# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-16 14:00:33
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-03 12:37:38
import tensorflow as tf
import numpy as np
import util.SEDWarp as SEDWarp
import matplotlib.pyplot as plt
import util.util as util
from scipy.misc import imresize
from Depth_Estim_Net import Depth_Estim_Net as DEN
import cv2

fnXML = "/users/visics/troussel/tmp/test_SIM3.xml"
bpath = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/RGB"
bpathDepth = "/esat/citrine/troussel/IROS/kinect/Kinect2/tmp_data/office4/depth"

h = 480
w = 640

config = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

rootFolder = "/esat/citrine/troussel/IROS/depth_estim/fixedRelu_L2/" 
weightsLoc = "%scheckpoint/" % rootFolder

net = DEN(weightsLoc, confFileName = config, training = False)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def warpTF(frame, depth, poseM):
	# Make placeholder graph
	inputGraph = tf.placeholder(tf.float32, shape = [2, h, w], name="RGB_in")
	depthGraph = tf.placeholder(tf.float32, shape = [2, h, w], name="depth_in")
	poseMGraph = tf.placeholder(tf.float32, shape = [2,4,4], name="poseM")
	# Build warping graph
	warpGraph = SEDWarp.warp_graph(depthGraph, inputGraph, poseMGraph)
	# Start tensorflow session
	with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
		# Run graph
		result = sess.run((warpGraph), feed_dict={inputGraph:frame, depthGraph: depth, poseMGraph: poseM})

	return result

def fprop(image):
	imageResized = imresize(image, (net.config["H"],net.config["W"]))
	imageResized = np.resize(imageResized, (1,net.config["H"],net.config["W"], 3))
	net.config["batchSize"] = 1
	depth = net.fprop(imageResized).squeeze() 
	return depth

def main():
	# Load single image-pair
	(poseM1, keyframe1, frame1, idName1) = SEDWarp.decode_xml(fnXML, 5, bpath)
	keyframe1 = imresize(keyframe1, (h,w))
	depth1 = fprop(keyframe1)
	depth2 = util.get_depth(idName1, bpathDepth)

	frame1 = imresize(rgb2gray(frame1), (h,w))
	depth1 = imresize(depth1, (h,w)) * 0.01
	depth2 = imresize(depth2, (h,w)) * 0.01
	# Warp image using tensorflow implementation
	frames = np.stack([frame1, frame1], axis=0)
	depths = np.stack([depth1, depth2], axis=0)
	poses = np.stack([poseM1, poseM1], axis=0)
	warpedTF = warpTF(frames, depths, poses)

	cannyIm = cv2.Canny(keyframe1.astype(np.uint8), 100, 200)
	cannyIm = imresize(cannyIm, (h,w))

	plt.subplot(131)
	plt.imshow(warpedTF[0]*0.5 + cannyIm*0.5, cmap = 'gray')
	plt.title("TensorFlow warping network")
	plt.subplot(132)
	plt.imshow(warpedTF[1]*0.5 + cannyIm*0.5, cmap = 'gray')
	plt.title("TensorFlow warping Kinect")
	plt.subplot(133)
	plt.imshow(np.abs(warpedTF[0] - rgb2gray(keyframe1)))
	plt.title("Difference keyframe - warped image")
	plt.show()

if __name__ == '__main__':
	main()