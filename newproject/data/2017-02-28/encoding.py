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
out_sparse1 = open('trainlist_no1.txt', 'r+')
'''
fname = open('both_list.txt', 'r+')

out_sparse2 = open('trainlist_no2.txt', 'r+')
out_sparse3 = open('trainlist_no3.txt', 'r+')
out_sparse4 = open('trainlist_no4.txt', 'r+')
out_test = open('test_list.txt', 'r+')
out_encoded1 = open('encoded_file.txt', 'w')
'''

'''
#Amino acid numbers assignment
aadict = [{'A': 1,'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
           'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,'Y': 20}]
'''




########################################Main function########################  
def encoding(file1):
  #file = open(file1, 'r+')
  #Amino acid numbers assignment
  aadict = {'A' : '10000000000000000000',
           'C' : '01000000000000000000',
           'D' : '00100000000000000000',
           'E' : '00010000000000000000',
           'F' : '00001000000000000000',
           'G' : '00000100000000000000',
           'H' : '00000010000000000000',
           'I' : '00000001000000000000',
           'K' : '00000000100000000000',
           'L' : '00000000010000000000',
           'M' : '00000000001000000000',
           'N' : '00000000000100000000',
           'P' : '00000000000010000000',
           'Q' : '00000000000001000000',
           'R' : '00000000000000100000',
           'S' : '00000000000000010000',
           'T' : '00000000000000001000',
           'V' : '00000000000000000100',
           'W' : '00000000000000000010',
           'Y' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

  top_dict = [{'I': 0, 'M': 1, 'O': 2}]

  sw = [1, 3, 5, 7, 9, 11, 13]
  


  #Creating Lists for Ids sequences and features
  aa_list = []
  feat_list = []
  #seqlist = []
  #sparse_list = []
  

  for counter, line in enumerate(file1):
    line = line.strip()
    line = line.split('\n')
    print("this is line:", line)
    for i in line:
      i = i.split()
      print('This is i:',i)
      if i in aadict.k:
        print('This is an AA:')
        aa_list.append(i)
      elif i in top_dict(item):
        print('This is an topology:')
        feat_list.append(i)
      
      
  	#line = line.split('\n')
  	#print(line)
  #	for i in range(len(line)):
  #		aacid = line[i]
  		#print(fit_transform(aadict,aacid))
  		#print('This is an AA', counter, aacid)
  
  
  #Closing the files which were opened
  #file.close()

  return

#fname = input("enter name of input file:")
print(encoding(out_sparse1))

#Closing the files which were opened

out_sparse1.close()


#file1.close()
#file2.close()
""" 
  out_sparse1.close()
  out_sparse2.close()
  out_sparse3.close()
  out_sparse4.close()
  out_test.close()
  """
