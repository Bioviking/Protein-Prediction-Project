#This script looks to conduct sparse encoding on the amino acid sequence



####################################Library imports################################

import numpy as np

first_dataset = open('../data/null_dataset/membrane-alpha.3line.txt', 'r+')
#nfile = open('../data/textfile/parsed/both_list.txt', 'r+')
nfile = open('../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')



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

top_dict_opp = {0 : 'I', 1 : 'M', 2 : 'O'}

wsize_list = [31, 5, 7, 13, 15, 17, 21, 25, 27, 35, 37]

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
    #ofile.close()
#    print(seq_list)
#    print(feat_list)
    return seq_list, feat_list

#######################################################Creating padding###############################################
def padding(link_list, wsize):
    
    pad =   [0] * 20 #[[0]*20] * sw
    wind_list= []
    wsize_list = [31, 5, 7, 13, 15, 17, 21, 25, 27, 35, 37]
    #wsize = int(input('Please confirm your window if not default of 3:'))
    odd = False
    while odd == False:
        for wsize in wsize_list:
            if wsize % 2 == 1:
                odd = True
                sw = int((wsize - 1)  / 2)
                print('Window size is :', wsize)
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

def encoding_file(nfile, wsize):
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
    
    wind_list = padding(link_list, wsize)   # Calling the padding and frame function
    top_list = [top_dict[aa] for pos in feat_list for aa in pos]   #Assigning the frames the features
#    print('this is toplist', top_list)
#    print('the is length list', len(top_list))

####Converting lists into an array##
    
    
#    X = np.array(wind_list)
#    y = np.array(top_list)
   
    
#    print(X.shape)
#   print(y.shape)
    
    svm_linear_learn(wind_list, top_list)
    #svm_RBF_learn(X, y) 
#    svm_learning(X, y) 
    
    return wind_list, top_list #, seq_list, feat_list, link_list
#    print(X)

###############################################Input my data for SVM###########################################

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def svm_linear_learn(wind_list, top_list):
   ####Converting lists into an array##
    X = np.array(wind_list)
    y = np.array(top_list)
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

    svc= SVC(kernel='linear', probability=True)

##Supervised learning
    svc.fit(X_train, y_train)

##Supervised Estimators

    y_pred = svc.predict(X_test) #p.random.random(())
    y_pred_prob = svc.predict_proba(X_test)
    feat_feature(y_pred, y_pred_prob, X_test)
          
##################################Evaluate my Model's Preformance########################

#Accuracy Score
#knn.score(X_test, y_test)
    from sklearn.metrics import accuracy_score
    print('This is the accuracy score:', accuracy_score(y_test, y_pred))

#Classification Report
    from sklearn.metrics import classification_report
    print('This is the classification report:', '\n' , classification_report(y_test,y_pred))

#ConfusionMatrix
    from sklearn.metrics import confusion_matrix
    print('This is the confusion matrix:', '\n' ,confusion_matrix(y_test, y_pred))

#Cross Validation
    score = cross_val_score(svc, X_train, y_train, cv=5)
    print("'This is the Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std()*2))
 
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
    print('This is the final structure prediction:')
    print(''.join(pre_list))
    print('This is the probability structure prediction:')
    #print(final_list) 
            
    return pre_list, final_list #X_test #final_pred

 
#################################################Confusion Matrix###################################################


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
    
    
for wsize in wsize_list:
    #ofile = open('../results/results' + str(wsize) + '.txt', 'w')
    nfile = open('../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')
    encoding_file(nfile, wsize)
    #svm_linear_learn(wind_list, top_list)     
#ofile.close()     
