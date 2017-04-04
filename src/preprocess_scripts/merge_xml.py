#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Tom Roussel
# @Date:   2017-04-04 09:26:59
# @Last Modified by:   Tom Roussel
# @Last Modified time: 2017-04-04 14:51:51

# This script merges several xml files into a single one

import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
import lxml.etree as etree

def main(fileList, outFile):
	top = ET.Element("Data")
	print("Looping over all input files")
	for index, fn in enumerate(fileList):
		print("Parsing and copying file %d" % index+1)
		tree = ET.parse(fn)
		root = tree.getroot()
		for frame in root:
			# Add frame to main tree
			top.append(frame)

	print("Writing to file")
	with open(outFile, 'w') as f:
		prettyRoot = etree.fromstring( ET.tostring(top, 'utf-8'))
		f.write(etree.tostring(prettyRoot, pretty_print = True))
	print("Done")


if __name__ == '__main__':
	# Parse commandline input
	parser = ArgumentParser(description="Merges xml files")
	parser.add_argument('--outFile','-o', help="Output file", required = True)
	parser.add_argument("files", nargs='+', help="Input xml files that need to be merged")
	args = parser.parse_args()
	main(args.files, args.outFile)