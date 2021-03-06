#This script looks to conduct sparse encoding on the amino acid sequence



####################################Library imports################################

import numpy as np

first_dataset = open('../data/null_dataset/membrane-alpha.3line.txt', 'r+')
#nfile = open('../data/textfile/parsed/both_list.txt', 'r+')
nfile = open('../data/textfile/cross_validated/temp_files/test_list.txt', 'r+')

##################################Creating Lists for Ids sequences and features##############

####Global Variables

aadict = {'A' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'C' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'D' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'E' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'F' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'G' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'H' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
          'I' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
          'K' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
          'L' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
          'M' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
          'N' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
          'P' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
          'Q' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
          'R' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
          'S' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
          'T' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
          'V' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
          'W' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
          'Y' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

top_dict = {'I': 0, 'M': 1, 'O': 2}

top_dict_inv = {'0':'I', '1':'M', '2':'O'}

#################0#######################Making seq and feat lists function######################## 
def encode_list(file1):
    seq_list = []
    feat_list = []
    ofile = file1
    
    for counter, line in enumerate(ofile):
        #print(line)
        line = line.strip('\n').split()
#        print(line)
        #print("this is line:", line)
        line = line[0] 
        
        if counter % 2 == 0:
#            print('This is a Match:', line)
            seq_list.append(line)
    
        else:
            #print('This is an topology:', line)
            feat_list.append(line)
    ofile.close()
#    print(seq_list)
#    print(feat_list)
    return seq_list, feat_list

#######################################################Creating padding###############################################
def padding(link_list):
    
    pad =   [0] * 20 #[[0]*20] * sw
    wind_list= []
    #sw = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 ,27, 33]
    wsize = int(input('Please confirm your window if not default of 3:'))
    odd = False
    while odd == False:
        if wsize % 2 == 1:
            odd = True
            sw = int((wsize - 1)  / 2)
            for pos in link_list:
                plen = len(pos)       
                for aa in range(plen):
                   
                    if aa < sw:
                        tempWin = pad*(sw-aa) + [i for am in pos[:(wsize-(sw-aa))] for i in am] 
                        wind_list.append(tempWin) 
                        
                    elif aa >= (plen - sw): #
                        tempWin = [i for am in pos[(aa-sw):plen] for i in am] + pad*(sw-((plen-1)-aa))
                        wind_list.append(tempWin)
                         
                    else:
                        tempWin = [i for am in pos[(aa-sw):(aa+1+sw)] for i in am]
                        wind_list.append(tempWin)
                        
                 
#       print(wind_list)
#       print(len(wind_list))
#       sys.exit(1)
#            print(wind_list)
#            print(len(wind_list))
            return wind_list
        else:
            wsize = int(input('Please enter an odd number or choose default 3:'))
    return

################################################Encoding function###################################################################

def encoding_file(nfile):
    file1 = nfile
    wind_list = []
    seq_list = []
    feat_list = []
    top_list = []
    link_list = []

    #Amino acid numbers assignment    
    seq_list, feat_list = encode_list(file1)
#    print(seq_list)
#    print(feat_list) 
    
    for counter, line in enumerate(seq_list):
        aa_list = []
        for aa in line:
            i = aadict[aa]

            aa_list.append(i)
        link_list.append(aa_list) 
    
    wind_list = padding(link_list)   # Calling the padding and frame function
    top_list = [top_dict[aa] for pos in feat_list for aa in pos]   #Assigning the frames the features
#    print('this is toplist', top_list)
#    print('the is length list', len(top_list))

####Converting lists into an array##
    
    
    X = np.array(wind_list)
    y = np.array(top_list)
   
    
    print(X.shape)
    print(y.shape)
    

    svm_RBF_learn(wind_list, top_list) 
#    svm_learning(X, y) 
    
    return wind_list, top_list
 
###############################################Input my data for SVM###########################################

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def svm_RBF_learn(wind_list, top_list):
        
    X = np.array(wind_list)
    y = np.array(top_list)
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #scaler = Normalizer().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators
    kernel_fit(X_train, y_train)
    #svc= SVC(kernel='rbf', probability=True)

##Supervised learning
    svc.fit(X_train, y_train)

##Supervised Estimators

    y_pred = svc.predict(X_test) #p.random.random(())
    y_pred_prob = svc.predict_proba(X_test)
    #feat_feature(y_pred, y_pred_prob, wind_list)        
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
    
    return   

#############Predicted features##################################   
def feat_feature(y_pred, y_pred_prob, X_test):    
    top_dict_opp = {0 : 'I', 1 : 'M', 2 : 'O'}
    y_pred = list(y_pred)
    #y_pred_prob = list(y_pred_prob)
    final_list = []
    pre_list = []
    for feat in range(len(X_test)):
        temp_feat = y_pred[feat]
        temp_prob = y_pred_prob[feat]
        #print(temp_feat)
        if temp_feat in top_dict_opp.keys():
            temp_feat = top_dict_opp[temp_feat]
            pre_list.append(temp_feat)
            final_list.append(temp_prob)
    print(''.join(pre_list))
    print(final_list) 
            
    return pre_list, final_list #X_test #final_pred


#######################################Tune_MY_Model##################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def kernel_fit(X, y):
    #X = X[y != 0, :20]
    #y = y[y != 0]
    

    print( len(X), len(y)) 
    n_sample = len(X)
    print('n_sample', n_sample)
    
    np.random.seed(0)
    order = np.random.permutation(n_sample)
    
    print(order)
    
    X = X[order]
    y = y[order].astype(np.float64)

    print(X, y)
    
    
    
    X_train = X[::n_sample]
    X_test = y[::n_sample]
    y_train = X[n_sample:]
    y_test = y[n_sample:]
    
    
    # fit the model
    print('hello kernel_fit')
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
