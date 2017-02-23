#####################################################3
#Creating a parsed file for the extraction of features for the original dataset.

#library imports
import os 
import numpy as np


# Load the data from file.
fname = open('~/Documents/Protein_Project/newproject/data/2017-02-21/membrane-alpha.3line.txt', 'r+')

#createing Lists for Ids sequences and features

idlist = []
seqlist = []
feat_list = []

for line in fname:
	print(line)
	line = line.strip('\n').split
	line = line[1]
	idlist.append(line)
	print(idlist[counter])
	
