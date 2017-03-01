#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:49:08 2017

@author: ryno
"""

from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)

X, y = iris.data, iris.target

clf.fit(X, y)
