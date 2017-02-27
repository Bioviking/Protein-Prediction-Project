#This script looks to conduct sparse encoding on the amino acid sequence


#creating class
#class sklearn.preprocessing



#library imports
from sklearn import preprocessing
import os 
import numpy as np
import scipy as sp
#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


#Hot encoding 
#enc = preprocessing.OneHotEncoder()
#enc.transform([[0, 1, 3]]).toarray()



# Load the data from file
fname = open('both_list.txt', 'r+')

out_sparse1 = open('trainlist_no1.txt', 'w')
out_sparse2 = open('trainlist_no2.txt', 'w')
out_sparse3 = open('trainlist_no3.txt', 'w')
out_sparse4 = open('trainlist_no4.txt', 'w')
out_test = open('test_list.txt', 'w')


#Amino acid numbers assignment
aadict = [{'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}]
topdict = [{'I': 0, 'M': 1, 'O': 3}]


#Creating my window triplet
#enc.fit([0,0,0],[1,0,3])


vect = DictVectorizer(sparse= False).fit(aadict)
print(vect.transform(aadict))
print ("feature names: %s" % vect.get_feature_names())


#Creating Lists for Ids sequences and features
aa_list = []
idlist = []
seqlist = []
sparse_list = []
for counter, line in enumerate(fname):
	line = line.strip()
	#line = line.split('\n')
	#print(line)
	for i in range(len(line)):
		aacid = line[i]
		#print(fit_transform(aadict,aacid))
		#print('This is an AA', counter, aacid)


#Closing the files which were opened
fname.close()
out_sparse.close()


