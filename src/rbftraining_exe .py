
#    print(X)

###############################################Input my data for SVM###########################################

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def svm_linear_learn(wind_list, top_list):
   
    X = np.array(wind_list)
    y = np.array(top_list)
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

    svc= SVC(kernel='rbf', probability=True)

##Supervised learning
    svc.fit(X_train, y_train)

##Supervised Estimators

    y_pred = svc.predict(X_test) #p.random.random(())
    y_pred_prob = svc.predict_proba(X_test)
    feat_feature(y_pred, y_pred_prob)        
##################################Evaluate my Model's Preformance########################

#Accuracy Score
#knn.score(X_test, y_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))

#Classification Report
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

#ConfusionMatrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

#Cross Validation
    scores = cross_val_score(svc, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
    

#############Predicted features##################################   

def feat_feature(y_pred, y_pred_prob):    
    
    print(y_pred, y_pred_prob)
    
    y_pred = list(y_pred)
    pre_list = []
    pre_list = [top_dict_inv[feat] for pos in len(y_pred) for feat in pos]   #Assigning the frames the features
#    for feat in y_pred:
#       if y_pred[feat] == top_dict_inv.keys[feat]:
#        feat = top_dict_inv.keys[feat]
#        pre_list.append(feat)
    final_pred= ''.join(pre_list)            
    print(len(final_pred))
    print(final_pred)    
    
#    print(pred_list)
        
    #return pred_list  
    return final_pred
##########################################################################################3
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

###########################Grid Search############################
from sklearn.grid_search import GridSearchCV
def search_grid
    params = {"n_neighbors":np.arange(1,3), "metrics": ["euclidean", "cityblock"]}
    grid = GridSearchCV(estimator=knn, param_grid=params)
    gird.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_estimator_.n_neighbors)
    return
########################Randomized Parameter Optimization################3

from sklearn.grid_search import RandomizedSearchCV

def rand_search
    params = {"n_neighbors":np.arange(1,5), "weights": ["uniform", "distance"]}
    rsearch = RandomizedSearchCV(estimator=knn, param_distributions=params, cv=4,n_iter=8, random_states=5)
    rsearch.fit(X_train, y_train)
    print(grid.best_score_)
    print(rsearch.best_score_)

def confusion_matrix(X, y)    
    
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    
    # We learn the digits on the first half of the digits
    classifier.fit(X[:n_samples / 2], y.target[:n_samples / 2])
    
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
encoding_file(nfile)

