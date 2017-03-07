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


out_sparse1 = '../data/textfile/cross_validated/testtrainlist_no1.txt'
out_formatted = '../data/textfile//encoded/formatted1.txt'

##################################Creating Lists for Ids sequences and features##############

########################################Making seq and feat lists function########################  

def encoding_list(file1):
    seq_list = []
    feat_list = []
    nfile = open(file1, 'r+')
    for counter, line in enumerate(nfile):
#        print(line)
        line = line.strip('\n').split()
#        print(line)
        #print("this is line:", line)
        line = line[0] 
        if counter % 2 == 0:
            #print('This is a Match:', line)
            seq_list.append(line)
    
        else:
            #print('This is an topology:', line)
            feat_list.append(line)
    nfile.close()
#    print(seq_list)
#    print(feat_list)
    return seq_list, feat_list



################################################Encoding function###################################################################

def encoding(file1, file2):
  
    #nfile = open(file1, 'r+')
  
    seq_list = []
    feat_list = []
    top_list = []
#    aa_list = []
    link_list = []
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
    seq_list, feat_list = encoding_list(file1)
#    print(seq_list)
#    print(feat_list) 
    
    for counter, line in enumerate(seq_list):
#        print('this is the aa seq list', line)
#        line = line.strip().split('\n')
#        print(line)
        aa_list = []
        for aa in line:
            i = aadict[aa]
#            print(i)
            aa_list.append(i)
        link_list.append(aa_list)   
#        print('this is the length of the inital aa list', len(aa_list))
#        print(aa_list)
#        print('this is the length of the final aa list', len(link_list))
#    print(link_list)  
    window_maker(link_list)
#        print(aa_list)
        
#        print(link_list)
#        break
              
#        print('\n')
#        print('\n')
#        print('\n')
          
#        link_list = link_list.
#        print(link_list)

##########IMPORTANT do not change#####################3
    for counter, line in enumerate(feat_list):
#        print('this is the topology list', line)
        for feat in line:
            i = top_dict[feat]
#            print('this in a single feat', i)
            top_list.append(i)
            

    print(top_list)

    ofile.close()



#######################################################Creating padding###############################################

def padding(link_list, sw):
   pad = [0] * 20   
   wind_list = []
   print('these are the variables')
   print('pad', pad)
   
   
   for pos in link_list:
       plen = len(pos)
       for aa in range(plen):
           if aa == 0:
               wind_list.append(pad*sw + pos[aa] + pos[aa + 1])
           elif aa == (plen - 1):
               wind_list.append(pos[aa - 1]+ pos[aa] + pad*sw)
           else:
               wind_list.append(pos[aa-sw]+ pos[aa] + pos[aa+sw]) 
   print(wind_list)
        
#        temp_list = []
#        wind_pad = []
#        pad = [pad] * sw
#        print('wind_pad', pad)
#        print('this is 1st wind_pad', pad)
#        temp_list = pad
#        print('this is temp list', temp_list)
#        for aa in range(len(pos)):
#            temp_list.append(pos[aa])                   
#        print('this is pad ', pad, 'this end')
#        temp_list.append(pad)
#        print('this is the temporay list', temp_list)
#        wind_list = temp_list
#   window_maker(temp_list, sw, size)
##################################Expanding the Window##################################
#
def window_maker(link_list):
    size = int(input('Please confirm your window if not default of 3:'))
    odd = False
    while odd == False:
        if size % 2 == 1:
            odd = True
            sw = int((size - 1)  / 2)
            padding(link_list, sw)
            return odd
        else:
            size = int(input('Please enter an odd number or choose default 3:'))
            
        
#    print('user input', size, 'sw', sw)
#    for pos in wind_list:
#        
#        pos = pos[0]
#        print(pos)
#        for da in range(len(temp_list)):
###           print(aa - sw)
#            win_list.append(temp_list[::size]) 
#        print('this is the final list', win_list)
#        print('end')
#            for aa in range(len(pos)):
#           print(aa, pos[aa])
#            print(aa, pos[aa-1:aa+1])
#            win_list.append(pos[aa-1:aa+2]) 


#    break
#            if aa == 0:
#                print('This is the first pos:', aa)
##               print(link_list[aa])
#                win_list.append(pad + pos[aa] + pos[aa + 1])
##                print('Pos 0:', win_list)
##            
#            elif aa == len(pos)-1:
#                print('This is the last pos:', aa)
#                win_list.append(pos[aa-1] + pos[aa] + pad)
##                print('pos end:', win_list)
#            else:
#                print(aa)
##                win_list.append(pos[aa-1] + pos[aa] + pos[aa+1] )
##                print(win_list)
#            print('end of aa:', win_list)
##        break
##        print('this is link_list', len(win_list))    
###                
##
#    




################################################Calling functions###############################

print(encoding(out_sparse1, out_formatted))

##################################Closing the files which were opened################################33


""" 
file1.close()
file2.close()

    out_sparse1.close()
out_formatted.close()
  out_sparse1.close()
  out_sparse2.close()
  out_sparse3.close()
  out_sparse4.close()
  out_test.close()
 """
