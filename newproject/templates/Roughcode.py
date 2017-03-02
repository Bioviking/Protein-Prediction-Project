#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:22:44 2017

@author: ryno
"""


########Rough pieces of code for other scripts


#feature extraction for sklearn svm

X= [{'A': 1, }

}]

from sklearn.feature_extraction import DictVectorizer
vect = DictVectorizer(sparse= False).fit(X)
print(vect.transform(X))
print ("feature names: %s" % vect.get_feature_names() )



'''	if counter % 3 == 0:
		line = line.strip('>')
		line = line.split('\n')
		line = line[0]
		idlist.append(line)	
		#print(line)
		out_ids.write(line + '\n')
		
'''

#??????  = np.array([aa * 20] * sw)


#Creating my window triplet
#enc.fit([0,0,0],[1,0,3])

'''
##DictVectorizer code
vect = DictVectorizer(sparse= False).fit(aadict)
print(vect.transform(aadict))
print ("feature names: %s" % vect.get_feature_names())
'''


###############################################From Encoding.py###################################################

#            print(aa_list)
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


      
  	#line = line.split('\n')
  	#print(line)
  #	for i in range(len(line)):
  #		aacid = line[i]
  		#print(fit_transform(aadict,aacid))
  		#print('This is an AA', counter, aacid)
  
  
  #Closing the files which were opened
  #file.close()
        

#fname = input("enter name of input file:")

