# This function looks to create the train and test set of the dataset once it has been parsed and prior to encoding



#library imports
#import os 
import numpy as np
#import scipy as sp
import sys 


# Load the data from file
fname = open('../data/textfile/parsed/both_list.txt', 'r+')
out_sparse1 = open('../data/textfile/cross_validated/trainlist_no1.txt', 'w')
out_sparse2 = open('../data/textfile/cross_validated/trainlist_no2.txt', 'w')
out_sparse3 = open('../data/textfile/cross_validated/trainlist_no3.txt', 'w')
out_sparse4 = open('../data/textfile/cross_validated/trainlist_no4.txt', 'w')
out_test = open('../data/textfile/cross_validated/test_list.txt', 'w')
#f2name = open('feat_list.txt', 'r+')
#out_sparse1 = open('../data/textfile/cross_validated/trainlist_no1.txt', 'w')
#out_sparse2 = open('../data/textfile/cross_validated/trainlist_no2.txt', 'w')
#out_sparse3 = open('../data/textfile/cross_validated/trainlist_no3.txt', 'w')
#out_sparse4 = open('../data/textfile/cross_validated/trainlist_no4.txt', 'w')
#out_test = open('../data/textfile/cross_validated/test_list.txt', 'w')

def protein_cross(file1):
    
    

    #List declaration
    trainlist = [1, 2, 3, 4, 5, 6, 7, 8]
    list1 = []
    list2 = []
    listA = []
    listB = []
    listC = []
    listD = []
    count = 1
    #def cross_valid(file1, file2):
    #print() 
    for counter, line in enumerate(fname):
      #line = line.strip()
    
      if count in trainlist:
        list1.append(line)
#        print("2nd count", count)
       
        if count in trainlist[0:2]:    
#            print("this is count 1", count)
            count += 1
            out_sparse1.write(line)
            listA = [line]

        elif count in trainlist[2:4]:
            out_sparse2.write(line)
            listB = [line]

            count += 1
        elif count in trainlist[4:6]:
            out_sparse3.write(line)
            listC = [line]

            count += 1
        elif count in trainlist[6:8]: 
            out_sparse4.write(line)
            listD = [line]
            count += 1
         
      elif count == 9:
        out_test.write(line)
        count += 1
        list2.append(line)
      elif count == 10 :
#        print("this is count 5", count)
#        print(counter)
        list2.append(line)
        out_test.write(line)
        count = 1
        
#    print("this list 1", len(list1))
#    print("this is list 2", len(list2))
    return listA, listB, listC, listD


#out_sparse1, out_sparse2, out_sparse3, out_sparse4
    out_sparse1.close()
    out_sparse2.close()
    out_sparse3.close()
    out_sparse4.close()
    out_test.close()
    

#list1, list2
    

#######################################################Window Cross Validation###################################33

#def window_cross()



#Calling the function
#print("this is the function", 

protein_cross(fname)
#window_cross()



#closing the files
fname.close()
#f2name.close()

#out_both.close()
