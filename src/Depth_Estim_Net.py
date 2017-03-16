# -*- coding: utf-8 -*-
# @Author: troussel
# @Date:   2017-02-03 15:40:55
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-16 11:03:29

import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm, max_pool2d, fully_connected
import yaml
import numpy as np
from math import floor

# TODO: Implement evaluation & fprop

class Depth_Estim_Net(object):
	def __init__(self, weightsLoc, summaryLocation = None, config=None, confFileName=None, training=True):
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
		self.weightsLoc = weightsLoc
		self.sess = None # Session when fpropping
		self.sumWriter = None

	def parse_config(self, conf):
		assert self._check_conf_dictionary(conf), "Configuration is invalid, parameters are missing"
		self.config = conf

	def parse_config_from_file(self, fn):
		"""
			Reads yaml file, fn being the path to that file
		"""
		with open(fn, 'r') as fid:
			conf = yaml.load(fid.read())
		# import pdb; pdb.set_trace()
		self.parse_config(conf)

	def dump_config_file(self, fn):
		"""
			Writes the current configuration file to disk
		"""
		with open(fn, 'w') as fid:
			yaml.dump(self.config, stream=fid, default_flow_style=False)

	def conv_layer(self, inGraph, outChannels, kernelSize, stride=1):
		# https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_#convolution2d
		# stride?
		# padding?
		# weights initialization
		return convolution2d(inputs = inGraph, num_outputs = outChannels, kernel_size = kernelSize, padding='VALID', stride=stride)

	def in_between_layers(self, inGraph, training, maxPool = True, scopeN ='inbetw'):
		"""
			This adds all the layers usually between the convolutions
			Batch norm, max pool, ...
		"""
		# Decay?
		# Other parameters?
		with tf.variable_scope(scopeN) as scope:
			graphTail = batch_norm(inputs = inGraph, decay = self.config["batchNormDecay"], is_training = training, reuse = not training ,updates_collections = None, scope = scope)

			if maxPool:
				# padding?
				graphTail = max_pool2d(inputs = graphTail, kernel_size = self.config["maxPoolKernel"], stride = self.config["maxPoolStride"], padding = "VALID")

			graphTail = tf.nn.relu(graphTail)
		return graphTail

	def build_graph(self, training):
		self.training = training
		# Input placeholder graph
		with tf.name_scope("Input"):
			self.inputGraph = tf.placeholder(tf.float32, shape = [self.config["batchSize"], self.config["H"], self.config["W"], 3], name="RGB_in")

		with tf.name_scope("Convolution_layers"):
			# Several layers convolution layers
			graphTail = self.conv_layer(self.inputGraph, outChannels = 96, kernelSize = 11, stride = 4)
			graphTail = self.in_between_layers(graphTail, training = training, scopeN = "NormMaxRelu1")

			graphTail = self.conv_layer(graphTail, outChannels = 256, kernelSize = 5)
			graphTail = self.in_between_layers(graphTail, training = training, scopeN = "NormMaxRelu2")

			graphTail = self.conv_layer(graphTail, outChannels = 384, kernelSize = 3)
			graphTail = self.in_between_layers(graphTail, maxPool = False, training = training, scopeN = "NormRelu1")
			
			graphTail = self.conv_layer(graphTail, outChannels = 384, kernelSize = 3, stride=2)
			graphTail = self.in_between_layers(graphTail, maxPool = False, training = training, scopeN = "NormRelu2")

			graphTail = self.conv_layer(graphTail, outChannels = 256, kernelSize = 3)

		# Output layers
		with tf.name_scope("Fully_connected"):
			# Flatten
			graphTail = tf.reshape(graphTail, [self.config["batchSize"], -1])
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

	def load_weights(self, sess, checkpoint, saver):
		newestWeights = tf.train.latest_checkpoint(checkpoint)
		print("Loading weights from checkpoint %s" % newestWeights)

		saver.restore(sess, newestWeights)

	def add_image_summary(self, images, name, batches, isRGB, width = -1, height = -1, reshape = False):
		# import pdb; pdb.set_trace()
		if reshape and isRGB:
			images = tf.reshape(images, [-1, height, width,  3])
		elif reshape and not isRGB:
			images = tf.reshape(images, [-1, height, width,  1])


		slicedImages = tf.slice(images, [0,0,0,0], [batches,-1,-1, -1])

		tf.summary.image(name, slicedImages)

	def summaries(self, training = True):
		# Images
		sumImageAmount = 3 if self.config["batchSize"] >= 3 else self.config["batchSize"]
		self.add_image_summary(self.fullGraph, "Depth Estimation", sumImageAmount, False, width = self.config["WOut"], height = self.config["HOut"], reshape = True)
		self.add_image_summary(self.inputGraph, "RGB", sumImageAmount, True)		

 		tf.summary.histogram("Depth", self.fullGraph)
		if training:
			self.add_image_summary(tf.reshape(self.gtDepthPlaceholder, [-1, self.config["HOut"], self.config["WOut"], 1]), "GT", 3, False)

		# Scalar
		if training:
			tf.summary.scalar("Loss", self.loss) 

	def train(self, trainingData, loadChkpt = False):
		"""
			Train the network using inputData. This should be a numpy array, [images x H x W x C].
			gtDepth [images x H x W]
			@trainingData: Is a generator function that outputs 2 variables, (in_rgb, in_depth)
		"""
		# Define optimizer
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config["learnRate"])
		print("Building graph")
		with tf.name_scope("Training"):
			self.fullGraph = self.build_graph(training = True)
			self.gtDepthPlaceholder = tf.placeholder(tf.float32, shape = [self.config["batchSize"], self.config["HOut"], self.config["WOut"]], name="depth_in")
			self.loss = self.add_l2_loss(self.fullGraph, self.gtDepthPlaceholder)
			
			# Add summaries
			self.summaries()
			# Add global step variable
			global_step = tf.Variable(0, name="global_step")
			increment_global_step_op = tf.assign(global_step, global_step+1)

			# Applying gradients
			grads = optimizer.compute_gradients(self.loss)
			trainOp = optimizer.apply_gradients(grads)
			sumOp = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()

		print("Starting tensorflow session")
		for key in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): print(key)
		with tf.Session() as sess:
			if not self.summaryLocation is None:
				self.sumWriter = tf.summary.FileWriter(self.summaryLocation, graph=sess.graph)
			saver = tf.train.Saver()
			idT = 0
			
			if loadChkpt:
				print("Loading weights from file")
				self.load_weights(sess, self.weightsLoc, saver)
			else:
				print("Initializing weights")
				sess.run(init_op)
			print("Done, running training ops")
			for in_rgb, in_depth in trainingData:

				self.debug_pre_run(global_step)
				_, currentLoss, summary, step = sess.run([trainOp, self.loss, sumOp, increment_global_step_op], feed_dict = {self.gtDepthPlaceholder:in_depth, self.inputGraph: in_rgb})
				print("Current loss is: %1.3f" % currentLoss)
				self.debug_post_run(global_step)

				if not self.summaryLocation is None:
					self.sumWriter.add_summary(summary, step)

				# Save weights every x steps, where x is given in the config
				if "saveInterval" in self.config.keys():
					if step % self.config["saveInterval"] == 0:
						saver.save(sess, self.weightsLoc, global_step=step)
					
			saver.save(sess, self.weightsLoc, global_step=step)

	def fprop(self, inData):
		"""
			Simple forward propagation. inData is formated like this: [images x H x W x C]
			Returns depth maps in the same way
		"""
		if self.sess is None:
			# Initialize everything
			self.fullGraph = self.build_graph(training = False)
			self.summaries(training = False)
			self.sumOp = tf.summary.merge_all()

			self.sess = tf.Session()
			saver = tf.train.Saver()
			self.load_weights(self.sess, self.weightsLoc, saver)

		if not self.summaryLocation is None and self.sumWriter is None:
			self.sumWriter = tf.summary.FileWriter(self.summaryLocation, graph=self.sess.graph)

		(out, summary) = self.sess.run([self.fullGraph, self.sumOp], feed_dict = {self.inputGraph : inData})

		if not self.sumWriter is None:
			self.sumWriter.add_summary(summary, 0)

		return out

	def add_l2_loss(self, inGraph, gtDepth):
		flatDepth = tf.reshape(gtDepth, shape = (self.config["batchSize"], self.config["HOut"] * self.config["WOut"]))
		return tf.nn.l2_loss(tf.subtract(inGraph, flatDepth))

	def _check_conf_dictionary(self, conf):
		"""
			Checks if the dictionary contains all the parameters (as keys) and returns true if they are all present.
		"""
		# Contains every parameter that is required in the config
		parameters_REQUIRED = ["batchSize", "H", "W", "HOut", "WOut", "batchNormDecay", "maxPoolKernel", "maxPoolStride"
						,"learnRate"] 
		parameters_OPTIONAL = ["saveInterval"]
		parameters = parameters_REQUIRED + parameters_OPTIONAL

		# Check for unknown parameters
		assert not any([not x in parameters for x in conf.keys()]), "Unknown parameter given"
		# Check if all parameters are present
		return all([x in conf.keys() for x in parameters_REQUIRED])

	def debug_pre_run(self, idT):
		"""
			Generic debug function, run every time before sess.run is called
		"""
		return

	def debug_post_run(self, idT):
		"""
			Generic debug function, run every time after sess.run is called
		"""
		return



