#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:49:08 2017

@author: ryno
"""

import sys
import numpy as np
import scipy
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

######################################Training and Test Data##############
def svm_learning(X, y):
       
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

    svc= SVC(kernel='linear')

##Supervised learning
    svc.fit(X_train, y_train)

##Supervised Estimators
    y_pred = svc.predict() #p.random.random(()

##################################Evaluate my Model's Preformance########################

#Accuracy Score
#knn.score(X_test, y_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)

#Classification Report
#from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))

#ConfusionMatrix
#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, y_pred))

#Cross Validation


    cross_val_score(svc, X_train, y_train, cv=5)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))



for line in sys.stdin:
    print(line)
    


svm_learning(X, y)


