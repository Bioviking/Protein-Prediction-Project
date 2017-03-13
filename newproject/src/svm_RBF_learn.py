#This script looks to conduct sparse encoding on the amino acid sequence


#creating class
#class sklearn.preprocessing



####################################Library imports################################
import sys
import encoding.py as X, y
#from sklearn import preprocessing
#import os 
import numpy as np
#import scipy as sp
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn import svm
from sklearn.svm import SVC

#out_X_array = open('../data/textfile/encoded/X_array.txt', 'r+')
#out_Y_array = open('../data/textfile/encoded/Y_array.txt', 'r+')

top_dict_inv = {'0':'I', '1':'M', '2':'O'}

#################################################################################
################################################################################
#"""
#def loading_input(X, Y)
#    X_array = []
#    Y_array = []
#    nfile = open(file1, 'r+')
#    
#    for line in nfile:
##        print(line)
#        line = line.strip('\n').split()
##        print(line)
#        #print("this is line:", line)
#        line = line[0] 
#        
#        if counter % 2 == 0:
##            print('This is a Match:', line)
#            seq_list.append(line)
#    
#        else:
#            #print('This is an topology:', line)
#            feat_list.append(line)
#    nfile.close()
##    print(seq_list)
##    print(feat_list)
#    return seq_list, feat_list
#"""
######################################Training and Test Data##############
def svm_RBF_learn(X, y):
       
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

    svc= SVC(kernel='RBF')

##Supervised learning
    svc.fit(X_train, y_train)

##Supervised Estimators
    y_pred = svc.predict(X_test) 
    for i in len(y_pred):
        if y_pred[i] == top_dict_inv.keys[i]:
            print(top_dict_inv.keys[i])
        
##################################Evaluate my Model's Preformance########################

#Accuracy Score
#knn.score(X_test, y_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)

#Classification Report
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

#ConfusionMatrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

#Cross Validation
    score = cross_val_score(svc, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std()*2))



#for line in sys.stdin:
#    print(line)
#    

















################################################Calling functions###############################
#loading_input(X, Y)

#encoding_list(file1)
svm_RBF_learn(X, y)
#training_svm(X, Y)
#padding(link_list)
##################################Closing the files which were opened################################33
#out_X_array.close()
#out_Y_array.close()

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
