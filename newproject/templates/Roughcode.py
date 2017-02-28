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

