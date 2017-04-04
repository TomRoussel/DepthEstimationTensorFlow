# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-04-03 13:33:58
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-04 16:46:24

import Depth_Estim_Net.Depth_Estim_Net
import tensorflow as tf
import util.SEDWarp as SEDWarp

# FIXME: untested!
# TODO: scale factor is not considered

class Slam_Loss_Net(Depth_Estim_Net):
	def train(self, trainingData, loadChkpt = False, lossFunc = None):
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

			self.loss = self.add_loss(self.fullGraph)

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
		with tf.Session(config = self.tfConfig) as sess:
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
			for in_rgb, poseM, warpFrames in trainingData:

				self.debug_pre_run(global_step)
				feed_dict = {self.inputGraph: in_rgb, self.poseMGraph: poseM, self.tFrame: warpFrames}
				_, currentLoss, summary, step = sess.run([trainOp, self.loss, sumOp, increment_global_step_op], feed_dict = feed_dict)
				print("Current loss is: %1.3f" % currentLoss)
				self.debug_post_run(global_step)

				if not self.summaryLocation is None:
					self.sumWriter.add_summary(summary, step)

				# Save weights every x steps, where x is given in the config
				if "saveInterval" in self.config.keys():
					if step % self.config["saveInterval"] == 0:
						saver.save(sess, self.weightsLoc, global_step=step)
					
			saver.save(sess, self.weightsLoc, global_step=step)

	def add_loss(self, inGraph):
		"""
			Constructs the warping loss
		"""
		with tf.name_scope("loss"):
			# Define input graphs for warping frame and pose matrix
			with tf.name_scope("GT"):
				self.poseMGraph = tf.placeholder(tf.float32, shape = (self.config["batchSize"],4,4))
				self.tFrame = tf.placeholder(tf.float32, shape = (self.config["batchSize"], self.config["H"], self.config["W"]),3)

			with tf.name_scope("Warping"):
				# Ugly oneliner that resized the depth image to the same size as the input frames
				depthResized = tf.squeeze(tf.image.resize_images(tf.reshape(depth, (self.config["batchSize"], self.config["HOut"], self.config["WOut"],1))), (self.config["batchSize"], self.config["H"], self.config["W"]))
				tFrameGray = tf.squeeze(tf.image.rgb_to_grayscale(self.tFrame))
				warped = SEDWarp.warp_graph(inGraph, tFrameGray, self.poseMGraph)
				oobPixels = warped == 0

			with tf.name_scope("L2"):
				tKeyFrame = tf.squeeze(tf.image.rgb_to_grayscale(self.inputGraph))
				diff = warped - tKeyFrame
				# Make sure out of bounds pixels are not considered
				diff *= tf.cast(tf.logical_not(oobPixels), tf.float32)
				loss = tf.nn.l2_loss(diff)

			return loss