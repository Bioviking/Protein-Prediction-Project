#####################################################3
#Creating a parsed file for the extraction of features for the original dataset.

#library imports
import os 
import numpy as np
import scipy as sp


# Load the data from file.
fname = open('membrane-alpha.3line.txt', 'r+')
out_ids = open('idlist.txt', 'w')
out_seq = open('seqlist.txt', 'w')
out_feat = open('feat_list.txt', 'w')

#creating Lists for Ids sequences and features
temp_list = []
idlist = []
seqlist = []
feat_list = []

#iterating through the file and separating the id label, sequences and topology

for counter, line in enumerate(fname):
	line = line.strip()
	if counter % 3 == 0:
		line = line.strip('>')
		line = line.split('\n')
		line = line[0]
		idlist.append(line)	
		#print(line)
		out_ids.write(line + '\n')
	elif counter % 3 == 1:	
		line = line.split('\n')
		line = line[0]
		#print('No 2', line)
		seqlist.append(line)
		out_seq.write(line + '\n')
	else:
		line = line.split('\n')
		line = line[0]
		#print('No 3', line)
		feat_list.append(line)
		out_feat.write(line + '\n')
		
			
#Printing the 3 lists to a text file 
'''print(out_id.write(idlist + '\n'))
print(out_seq.write(seqlist + '\n'))
print(out_feat.write(feat_list + '\n'))
'''	
		
#Closing the files which were opened


fname.close()
out_ids.close()
out_seq.close()
out_feat.close()
	
