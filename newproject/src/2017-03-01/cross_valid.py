# This function looks to create the train and test set of the dataset once it has been parsed and prior to encoding



#library imports
#import os 
#import numpy as np
#import scipy as sp



# Load the data from file
fname = open('both_list.txt', 'r+')
#f2name = open('feat_list.txt', 'r+')
out_sparse1 = open('trainlist_no1.txt', 'w')
out_sparse2 = open('trainlist_no2.txt', 'w')
out_sparse3 = open('trainlist_no3.txt', 'w')
out_sparse4 = open('trainlist_no4.txt', 'w')
out_test = open('test_list.txt', 'w')

#List declaration
trainlist = [1, 2, 3, 4, 5, 6, 7, 8]
list1 = []
list2 = []
count = 1
#def cross_valid(file1, file2):
#print() 
for counter, line in enumerate(fname):
  #line = line.strip()

  if count in trainlist:
    list1.append(line)
    print("2nd count", count)
   
    if count in trainlist[0:2]:    
        print("this is count 1", count)
        count += 1
        out_sparse1.write(line)
    elif count in trainlist[2:4]:
        out_sparse2.write(line)
        count += 1
    elif count in trainlist[4:6]:
        out_sparse3.write(line)
        count += 1
    elif count in trainlist[6:8]: 
        out_sparse4.write(line)
        count += 1
     
  elif count == 9:
    out_test.write(line)
    count += 1
    list2.append(line)
  elif count == 10 :
    print("this is count 5", count)
    print(counter)
    list2.append(line)
    out_test.write(line)
    count = 1
    
print("this list 1", len(list1))
print("this is list 2", len(list2))
    

#Calling the function
#print("this is the function", cross_valid(fname, out_sparse))

#closing the files
fname.close()
#f2name.close()
out_sparse1.close()
out_sparse2.close()
out_sparse3.close()
out_sparse4.close()
out_test.close()
#out_both.close()
