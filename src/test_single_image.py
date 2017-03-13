# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-09 16:04:29
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-10 10:56:55
from Depth_Estim_Net import Depth_Estim_Net as DEN
import numpy as np
import PIL.Image as Im
import matplotlib.pyplot as plt
from scipy.misc import imresize

imLoc = "/esat/citrine/tmp/troussel/IROS/kinect/Kinect2/kinData/office4/RGB/000168.png"

config = "/users/visics/troussel/Tensor_Workspace/Python_Code/depth_estim/conf/init.yaml"

rootFolder = "/esat/citrine/tmp/troussel/IROS/depth_estim/NYU_BN_scope/" 
weightsLoc = "%scheckpoint/" % rootFolder


def main():
	# Load image
	inIm = np.asarray(Im.open(imLoc))
	# Reshape
	inIm = imresize(inIm, (240,320))
	inIm = np.reshape(inIm, (1, inIm.shape[0], inIm.shape[1], inIm.shape[2]))
	# Remove last channel (not used)
	inIm = inIm[:,:,:,0:3]

	# Load Network
	net = DEN(weightsLoc, summaryLocation = "/users/visics/troussel/tmp/sum/", confFileName = config, training = False)
	net.config["batchSize"] = 1
	# Frop
	depth = net.fprop(inIm)
	print(depth.dtype)
	# Show
	plt.subplot(121)
	plt.imshow(inIm[0,:,:,:])
	plt.subplot(122)
	plt.imshow(np.reshape(depth, (55,74)))
	plt.colorbar()
	plt.show()

if __name__ == '__main__':
	main()
