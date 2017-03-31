# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-03-31 09:42:09
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-03-31 10:23:52

import os
import tensorflow as tf
import shlex
from subprocess import call

# NOTE: This will not work when using gcc 4 or older
#       Remove "-D_GLIBCXX_USE_CXX11_ABI=0" from cmd if this is the case
def compile_shared_library(fnCC, fnSO):
	"""
		Compiles a set of ops
		@fnCC: c++ file containing the source code
		@fnSO: path to the output shared library
	""" 
	print("Could not find op library, compiling...")
	includePath = tf.sysconfig.get_include()
	cmd = "g++ -std=c++11 -shared %s -o %s -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I %s" % (fnCC, fnSO, includePath)
	args = shlex.split(cmd)
	ret = call(args)
	if not ret == 0:
		print("Compiling failed!")
		raise ImportError
	else:
		print("Compilation succesfull")


path = os.path.dirname(os.path.abspath(__file__))
fnZeroOut = "%s/%s" % (path, "zero_out_two.cc")
fnZeroOutSO = "%s/%s" % (path, "zero_out_two.so")
# Check if .so file exists
if not os.path.isfile(fnZeroOutSO):
	compile_shared_library(fnZeroOut, fnZeroOutSO)

ZeroOutOps = tf.load_op_library(fnZeroOutSO)
