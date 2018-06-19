#####################################################3
#Creating a parsed file for the extraction of features for the original dataset.

#library imports
import os 
import numpy as np
import scipy as sp


# Load the data from file.
fname = open('membrane-alpha.3line.txt', 'r+')
out_ids = open('id_list.txt', 'w')
out_ids_seq = open('id_seq_list.fasta', 'w')
out_seq = open('seqlist.txt', 'w')
out_ids_feat = open('id_feat_list.txt', 'w')
out_feat = open('feat_list.txt', 'w')

out_both = open('both_list.txt', 'w')


#creating Lists for Ids sequences and features
main_dict = {}
temp_list = []
idlist = []
seqlist = []
feat_list = []

#iterating through the file and separating the id label, sequences and topology

for counter, line in enumerate(fname):
  line = line.strip()
  
  if counter % 3 == 0:
    line = line.strip()
    line = line.split('\n')
    print(line)
    line = line[0]
    print(line)
    
    idlist.append(line)
    keyA = idlist(line)
    #main_dict[keyA] += keyA 
    #print(line)
    out_ids.write(line + '\n')
    out_ids_seq.write(line + '\n')
    out_ids_feat.write(line + '\n')
  
  
  
  ######Sequence#################33333  
  elif counter % 3 == 1:
    line = line.split('\n')
    line  = line[0]
    print(line)
	#print('No 2', line)
    seqlist.append(line)
    main_dict[]
    out_seq.write(line + '\n')
    out_ids_seq.write(line + '\n')
    out_both.write(line + '\n')
  
  ######Topology###
  else:
    line = line.split('\n')
    line = line[0]
	#print('No 3', line)
    feat_list.append(line)
    out_feat.write(line + '\n')
    out_ids_feat.write(line + '\n')
    out_both.write(line + '\n')
  			
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
out_both.close()
