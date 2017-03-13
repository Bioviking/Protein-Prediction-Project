#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:39:36 2017

@author: ryno
"""

####################################Library imports################################
# Import datasets, classifiers and performance metrics
# Standard scientific Python imports
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import svm, metrics, preprocessing, neighbors
from sklearn.metrics import accuracy_score

######################################Training and Test Data##############
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

##################################PREprocessing############################

############################Standardization#############################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)


########################################Normalization#################################
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)


###################################Creating my Model##############################3

#Supervised Learning Estimators
from sklearn.svm import SVC
svc= SVC(kernel='linear')

#Supervised learning
svc.fit(X_train, y_train)

#Supervised Estimators
y_pred = svc.predict(np.random.random((2, 5))

##################################Evaluate my Model's Preformance########################

#Accuracy Score
#knn.score(y_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#ConfusionMatrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

#Cross Validation

from sklearn.cross_validation import cross_val_score
score = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

#######################################Tune_MY_Model##################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def kernel_fit(X_train, X_test, y_train, y_test)
    # fit the model
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        clf = svm.SVC(kernel=kernel, gamma=10)
        clf.fit(X_train, y_train)

        plt.figure(fig_num)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

        # Circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

        plt.title(kernel)
    plt.show()
    return


###########################Grid Search###########################3#
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors":np.arange(1,3), "metrics": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn, param_grid=params)
gird.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

########################Randomized Parameter Optimization################3

from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors":np.arange(1,5), "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn, param_distributions=params, cv=4,n_iter=8, random_states=5)
rsearch.fit(X_train, y_train)
print(grid.best_score_)
print(rsearch.best_score_)


#########################################Frame - Cross-validation-train_test_split################    
def svm_cross_valid(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf = svm.LinearSVC(class_weight = 'balanced', C=1)
    scores = cross_val_score(clf, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
    print(scores)
    


    clf = svm.LinearSVC()    #class_weight = 'balanced', C=1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = clf.fit(X_train, Y_train)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))
    return X, Y

########################3Train SVM#####################33
def training_svm(X, Y):
    return X, Y
    ########################################Predictor#######################3
    #     clf.predict(digits.data[-1:])
    
    #
    #import pickle
    #    s = pickle.dumps(clf)
    #    clf2 = pickle.loads(s)
    #    clf2.predict(X[0:1])
    #
    #    y[0]
    
    
#################################Confusion matrix#############################    
def confusion_matrix(X, Y)    
    
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    
    # We learn the digits on the first half of the digits
    classifier.fit(X[:n_samples / 2], Y.target[:n_samples / 2])
    
    # Now predict the value of the digit on the second half:
    expected = Y.target[n_samples / 2:]
    predicted = classifier.predict(X[n_samples / 2:])
    
    
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    
    images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)
    
    plt.show()



from sklearn import svm

clf = svm.SVC(gamma=0.001, C=5.)

X, y = iris.data, iris.target

clf.fit(X, y)





svm_cross_valid(X, Y):
training_svm(X, Y)

confusion_matrix(X, Y) 
