#This script looks to conduct sparse encoding on the amino acid sequence


#creating class
#class sklearn.preprocessing



####################################Library imports################################
import sys
#from sklearn import preprocessing
#import os 
import numpy as np
#import scipy as sp
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm

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

out_sparse1 = '../data/textfile/both_list.txt'
#out_sparse1 = '../data/textfile/both_list.txt'
#out_sparse1 = '../data/textfile/cross_validated/membrane-alpha.3line.txt'
out_formatted = '../data/textfile//encoded/formatted1.txt'

##################################Creating Lists for Ids sequences and features##############

####Global Varial'bles 
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
#            print('This is a Match:', line)
            seq_list.append(line)
    
        else:
            #print('This is an topology:', line)
            feat_list.append(line)
    nfile.close()
#    print(seq_list)
#    print(feat_list)
    return seq_list, feat_list

################################################Encoding function###################################################################

def encoding(file1): #### possible output file - , file2
  
    #nfile = open(file1, 'r+')
    wind_list = []
    seq_list = []
    feat_list = []
    top_list = []
#    aa_list = []
    link_list = []
#    ofile = open(file2, 'w')
    #Amino acid numbers assignment


    #sw = [1, 3, 5, 7, 9, 11, 13]
    seq_list, feat_list = encoding_list(file1)
#    print(seq_list)
#    print(feat_list) 
    
    for counter, line in enumerate(seq_list):
        aa_list = []
        for aa in line:
            i = aadict[aa]

            aa_list.append(i)
        link_list.append(aa_list) 
    
    wind_list = padding(link_list)   # Calling the padding and frame function
    top_list = [top_dict[aa] for pos in feat_list for aa in pos]   #Assigning the frames the features
#    print('this is toplist', top_list)
#    print('the is length list', len(top_list))


##########################Converting lists into an array#########################
    X = np.array(wind_list)
    Y = np.array(top_list)
    
    print(X.shape)
    print(Y.shape)
#    print(X)
    
#########################################Frame Cross-validation################    
    
#    clf = svm.LinearSVC(class_weight = 'balanced', C=1)
#    scores = cross_val_score(clf, X, Y, cv=5)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
#    print(scores)

########################3Train SVM#####################33
    clf = svm.LinearSVC()    #class_weight = 'balanced', C=1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = clf.fit(X_train, Y_train)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))
    
    ########################################Predictor#######################3
    #     clf.predict(digits.data[-1:])
    #    ofile.close()
    #
    #import pickle
    #    s = pickle.dumps(clf)
    #    clf2 = pickle.loads(s)
    #    clf2.predict(X[0:1])
    #
    #    y[0]


#######################################################Creating padding###############################################

def padding(link_list):
    pad =   [0] * 20 #[[0]*20] * sw
    wind_list= []
    wsize = int(input('Please confirm your window if not default of 3:'))
    odd = False
    while odd == False:
        if wsize % 2 == 1:
            odd = True
            sw = int((wsize - 1)  / 2)
            for pos in link_list:
                plen = len(pos)       
                for aa in range(plen):
                   
                    if aa < sw:
                        tempWin = pad*(sw-aa) + [i for am in pos[:(wsize-(sw-aa))] for i in am] 
                        wind_list.append(tempWin)   
                    elif aa >= (plen - sw): #
                        tempWin = [i for am in pos[(aa-sw):plen] for i in am] + pad*(sw-((plen-1)-aa))
                        wind_list.append(tempWin)
                    else:
                        tempWin = [i for am in pos[(aa-sw):(aa+1+sw)] for i in am]
                        wind_list.append(tempWin)
 
#       print(wind_list)
#       print(len(wind_list))
#       sys.exit(1)
#            print(wind_list)
#            print(len(wind_list))
            return wind_list
        else:
            wsize = int(input('Please enter an odd number or choose default 3:'))
    
#    sys.exit(1)        


################################################Calling functions###############################

encoding(out_sparse1)   #Possible out file - , out_formatted

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
