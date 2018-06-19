#This script looks to conduct sparse encoding on the amino acid sequence


#creating class
#class sklearn.preprocessing



####################################Library imports################################
#from sklearn import preprocessing
#import os 
#import numpy as np
#import scipy as sp
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_extraction import DictVectorizer


###################################Hot encoding##################################33 
#enc = preprocessing.OneHotEncoder()
#enc.transform([[0, 1, 3]]).toarray()
####################################################################################




##################Extra###################################3

#fname = open('both_list.txt', 'r+')
#
#out_sparse2 = open('trainlist_no2.txt', 'r+')
#out_sparse3 = open('trainlist_no3.txt', 'r+')
#out_sparse4 = open('trainlist_no4.txt', 'r+')
#out_test = open('test_list.txt', 'r+')
#out_encoded1 = open('encoded_file.txt', 'w')
#
##Amino acid numbers assignment
#aadict = [{'A': 1,'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
#           'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,'Y': 20}]


##################################### Load the data from file############################


out_sparse1 = 'testtrainlist_no1.txt'
out_formatted = 'formatted1.txt'

##################################Creating Lists for Ids sequences and features##############

########################################Main function########################  

def encoding_list(file1):
    seq_list = []
    feat_list = []
    nfile = open(file1, 'r+')
    for counter, line in enumerate(nfile):
        line = line.strip()
        line = line.split('\n')
        #print("this is line:", line)
        line = line[0] 
        if counter % 2 == 0:
            #print('This is a Match:', line)
            seq_list.append(line)
    
        else:
            #print('This is an topology:', line)
            feat_list.append(line)
    nfile.close()
    #print(seq_list)
    #print(feat_list)
    return seq_list, feat_list












def encoding(file1, file2):
  
    #nfile = open(file1, 'r+')
  
    list1 = []
    list2 = []
    link_list = []
    aa_list = []
    ofile = open(file2, 'w')
    #Amino acid numbers assignment
    aadict = {'A' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'C' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'D' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'E' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'F' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'G' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'H' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'I' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                'K' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                'L' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                'M' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                'N' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                'P' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                'Q' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                'R' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                'S' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                'T' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                'V' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                'W' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                'Y' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

    top_dict = {'I': 0, 'M': 1, 'O': 2}

    #sw = [1, 3, 5, 7, 9, 11, 13]
    list1, list2 = encoding_list(file1)
#    print(list1)
#    print(list2)

#    
    for counter, line in enumerate(list1):
        
        line = line.strip().split('\n')
        print(line)
        for aa in line:
            aa = aadict[aa]
            print(aa)
        #        line = line[counter]
#            aa_list.append(aadict[aa])
#            print(aa_list)
#            print(len(aa_list)) 
####            
#                print(aa, aadict[aa]) 
#                break
        
#        for counter, line2 in enumerate(list2):
#            line = line[0]
#            print(line, line2)
#            
#            for feat in line2:
#                print(feat)
#            print("this is line:", line)
            
#            line2 = line2[0]
            
#            print(counter)
    #        print("this is line:", line)
#        
#            aa = aa.split(' ')
#            print(aa)
##            aa = aa[0]
#            
#                line = line.strip().split('\n')
#                print(counter)
#                print('this is feat', line)
#            break
            
            
            
##                print(line)
#                line = line[0]
##                print(counter, line)
#    #            print('This is aa:', aa, )
#                break
#    #        print("this is line:", line)
#    #        print('this is a feat', line)
#                for counter in line:
#                    print(counter)
##                    feat = feat.split('')
#                    
##                    feat = feat[0]
#                    #print(feat)
##                    print('this is both:', top_dict[feat], aadict[aa])
#                    break
#     for counter in link_list:
#        print(counter)
    
#        link_list.append()   
        
#            print('This is aa:', feat, top_dict[feat])
#            print('Another the match ', top_dict[aa], aadict[aa])
#              ofile.write(aa, aadict[aa])
##            for aa in line:
#                i = i.split(' ')
#                i = i[0]
#                print('This is feat', aa)
#          aa = aa[0]
#          print('Another the match ', aa, top_dict[aa])
#          ofile.write(aa, aadict[aa])

    ofile.close()
  
     
      
  	#line = line.split('\n')
  	#print(line)
  #	for i in range(len(line)):
  #		aacid = line[i]
  		#print(fit_transform(aadict,aacid))
  		#print('This is an AA', counter, aacid)
  
  
  #Closing the files which were opened
  #file.close()
        

#fname = input("enter name of input file:")



################################################Calling functions###############################

print(encoding(out_sparse1, out_formatted))

##################################Closing the files which were opened################################33



#file1.close()
#file2.close()
""" 
    out_sparse1.close()
out_formatted.close()
  out_sparse1.close()
  out_sparse2.close()
  out_sparse3.close()
  out_sparse4.close()
  out_test.close()
  """
