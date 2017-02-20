# -*- coding: utf-8 -*-
# @Author: troussel
# @Date:   2017-02-03 15:40:55
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-02-20 11:02:20

import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected
import yaml
import numpy as np

# FIXME: Test graph
# FIXME: Test summary
# FIXME: Test config file I/O
# FIXME: Test loss function
# TODO: Implement evaluation & fprop


class Depth_Estim_Net(object):
	def __init__(self, summaryLocation, weightsLoc, config=None, confFileName=None, training=True):
		"""
			@config: 		A dictionary containing all the parameters for the network. Will throw an exception if there are parameters missing.
			@confFileName: 	Path to a yaml file containing the parameters of the network. Only this parameter or the config parameter can
							be set, not both.
			@training: 		Set this to true of the network is to be trained
			
		"""
		assert config is None or confFileName is None, "Both config and confFileName parameters are set, set one or the other."
		assert config is not None or confFileName is not None, "No configuration specified"

		if config is not None:
			self.parse_config(config)
		elif confFileName is not None:
			self.parse_config_from_file(confFileName)

		self.summaryLocation = summaryLocation
		self.weightsLoc = "%s/1.ckpt" % weightsLoc

	def parse_config(self, conf):
		assert _check_conf_dictionary(conf), "Configuration is invalid, parameters are missing"
		self.config = conf

	def parse_config_from_file(self, fn):
		"""
			Reads yaml file, fn being the path to that file
		"""
		conf = yaml.load(fn)
		assert _check_conf_dictionary(conf), "Configuration is invalid, parameters are missing"
		self.config = conf

	def dump_config_file(self, fn):
		"""
			Writes the current configuration file to disk
		"""
		with open(fn, 'w') as fid:
			yaml.dump(self.config, stream=fid, default_flow_style=False)

	def conv_layer(self, inGraph, outChannels, kernelSize):
		# https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_#convolution2d
		# stride?
		# padding?
		# weights initialization
		return convolution2d(inputs = inGraph, num_outputs = outChannels, kernel_size = kernelSize)

	def in_between_layers(self, inGraph, training, maxPool = True):
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
		self.training = training
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
			graphTail = self.in_between_layers(graphTail, maxPool = False, training = training)
			
			graphTail = self.conv_layer(graphTail, outChannels = 384, kernelSize = 3)
			graphTail = self.in_between_layers(graphTail, maxPool = False, training = training)

			graphTail = self.conv_layer(graphTail, outChannels = 256, kernelSize = 3)

		# Output layers
		with tf.name_scope("Fully_connected"):
			graphTail = fully_connected(inputs = graphTail, num_outputs = 4096)
			graphTail = fully_connected(inputs = graphTail, num_outputs = self.config["HOut"] * self.config["WOut"])

		return graphTail

	def train_dataset(self, rgb_in, depth_in):
		"""
			Calls the train function internally and constructs the generator function
			Expects data with [image x H x W x C]
		"""
		def generator(rgb_in, depth_in, batchSize):
			batches = int(rgb_in.shape(0)/batchSize)
			for x in xrange(batches):
				rgbBatch = np.asarray(rgb_in[x*batchSize:(x+1)*batchSize,:,:,:])
				depthBatch = np.asarray(depth_in[x*batchSize:(x+1)*batchSize,:,:])
				yield (rgbBatch, depthBatch)

		batchSize = self.config["batchSize"]
		self.train(generator(rgb_in, depth_in, batchSize))

	def load_weights(self, sess, checkpoint, training = False):
		print("Loading weights from checkpoint %s" % checkpoint)
		saver = tf.train.Saver()
		saver.restore(sess, checkpoint)


	def train(self, trainingData):
		"""
			Train the network using inputData. This should be a numpy array, [images x H x W x C].
			gtDepth [images x H x W]
			@trainingData: Is a generator function that outputs 2 variables, (in_rgb, in_depth)
		"""
		# Define optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learnRate"])
		print("Building graph")
		with tf.name_scope("Training"):
			self.fullGraph = build_graph(training = True)
			gtDepthPlaceholder = tf.placeholder(tf.float32, shape = [self.config["batchSize"], self.config["HOut"], self.config["WOut"]], name="depth_in")
			loss = add_l2_loss(self.fullGraph, gtDepthPlaceholder)
			tf.scalar_summary(loss)

			# Applying gradients
			grads = optimizer.compute_gradients(loss)
			trainOp = grads.apply_gradients(grads)
			sumOp = tf.merge_all_summaries()
			init_op = tf.global_variables_initializer()

		print("Starting tensorflow session")

		with tf.Session() as sess:
			sumWriter = tf.train.SummaryWriter(self.summaryLocation, graph=sess.graph)
			saver = tf.train.Saver()
			idT = 0
			
			print("Initializing weights")
			sess.run(init_op)
			print("Done, running training ops")
			for in_rgb, in_depth in trainingData:
				_, currentLoss, summary = sess.run([trainOp, loss, sumOp], feed_dict = {"Training/depth_in:0":in_depth, "Input/RGB_in:0": in_rgb})
				print("Current loss is: %1.3f" % currentLoss)
				
				sumWriter.add_summary(summary, idT)
				idT += 1

			saver.write(self.weightsLoc)


	def add_l2_loss(self, inGraph, gtDepth):
		flatDepth = tf.reshape(gtDepth, shape = (self.config["batchSize"], self.config["HOut"] * self.config["WOut"]))
		return tf.nn.l2_loss(tf.sub(self.fullGraph, flatDepth))

	def _check_conf_dictionary(self, conf):
		"""
			Checks if the dictionary contains all the parameters (as keys) and returns true if they are all present.
		"""
		parameters = ["batchSize", "H", "W", "HOut", "WOut", "batchNormDecay", "maxPoolKernel", "maxPoolStride"
						,"learnRate"]

		# Check for unknown parameters
		assert not any([not x in parameters for x in conf.keys()]), "Unknown parameter given"
		# Check if all parameters are present
		return all([x in conf.keys() for x in parameters])