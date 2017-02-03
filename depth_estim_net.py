# -*- coding: utf-8 -*-
# @Author: troussel
# @Date:   2017-02-03 15:40:55
# @Last Modified by:   troussel
# @Last Modified time: 2017-02-03 18:14:57

import tensorflow as tf
from tf.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected

class Depth_Estim_Net(object):
	def __init__(self, config, training=True):
		"""
			For now not sure what config will be. Most likely dictionary a dictionary for all values.
			
		"""
		parse_config(config)
		# The basegraph should be the 
		self.baseGraph = build_graph(training = training)

	def conv_layer(self, inGraph, outChannels, kernelSize):
		# https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_#convolution2d
		# stride?
		# padding?
		# weights initialization
		return convolution2d(inputs = inGraph, num_outputs = outChannels, kernel_size = kernelSize)

	def in_between_layers(self, inGraph, maxPool = True, training):
		"""
			This adds all the layers usually between the convolutions
			Batch norm, max pool, ...
		"""
		# Decay?
		# Other parameters?
		graphTail = batch_norm(inputs = inGraph, decay = self.config["batchNormDecay"], is_training = training)

		if maxPool:
			# padding?
			graphTail = max_pool2d(inputs = graphTail, kernel_size = self.config["maxPoolKernel"], stride = self.config["maxPoolStride"], padding = "SAME")

		graphTail = tf.nn.relu(graphTail)
		return graphTail


	def build_graph(self, training):
		# Input placeholder graph
		with tf.name_scope("Input"):
			inputGraph = tf.placeholder(tf.int, shape = [self.config["batchSize"], self.config["H"], self.config["W"], 3], name="RGB_in")

		with tf.name_scope("Convolution_layers"):
			# Several layers convolution layers
			graphTail = self.conv_layer(inputGraph, outChannels = 96, kernelSize = 11)
			graphTail = self.in_between_layers(graphTail, training = training)

			graphTail = self.conv_layer(graphTail, outChannels = 256, kernelSize = 5)
			graphTail = self.in_between_layers(graphTail, training = training)

			graphTail = self.conv_layer(graphTail, outChannels = 384, kernelSize = 3)
			graphTail = self.in_between_layers(graphTail, maxPool = False,training = training)
			
			graphTail = self.conv_layer(graphTail, outChannels = 384, kernelSize = 3)
			graphTail = self.in_between_layers(graphTail, maxPool = False,training = training)

			graphTail = self.conv_layer(graphTail, outChannels = 256, kernelSize = 3)

		# Output layers
		with tf.name_scope("Fully_connected"):
			graphTail = fully_connected(inputs = graphTail, num_outputs = 4096)
			graphTail = fully_connected(inputs = graphTail, num_outputs = 4070)

		return graphTail
		
	def parse_config(self, conf):
		"""
			This method will take parse the config parameter and set up all of the required parameters 
			for the network.
		"""
		raise NotImplementedError("Configuration still needs to be implemented.")