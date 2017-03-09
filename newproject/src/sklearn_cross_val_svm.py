#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:36:26 2017

@author: ryno
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.LinearSVC(class_weight = 'balanced', C = 1)
scores = cross_valscore(clf, X, Y, cv=5)
print(scores)

########################3Train SVM#####################33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_stat=0)
clf = clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))