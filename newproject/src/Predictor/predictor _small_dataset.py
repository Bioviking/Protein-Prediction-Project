#This script looks to conduct sparse encoding on the amino acid sequence



####################################Library imports################################

import numpy as np

#first_dataset = open('../../data/null_dataset/membrane-alpha.3line.txt', 'r+')
#nfile = open('../../data/textfile/parsed/both_list.txt', 'r+')
nfile = open('../../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')


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
    #sw = [3, 5, 7, 9, 11, 13] #Sliding window options
    #wsize = int(input('Please confirm your window if not default of 3:'))
    wsize = 3
    if wsize == 0:    
        odd = False
        while odd == False:
            if wsize % 2 == 1:
                odd = True
                sw = int((wsize - 1)  / 2)
                print(link_list)
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
                #return wind_list
    else:
        if wsize % 2 == 1:
                odd = True
                sw = int((wsize - 1)  / 2)
                print(link_list)
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
        
        
        
    return wind_list

################################################Encoding function###################################################################

def encoding_file(nfile, pfile):
    file1 = nfile
    wind_list = []
    seq_list = []
    feat_list = []
    top_list = []
    link_list = []
    window = 0
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
    
 #   confusion_matrix(wind_list, top_list)
    svm_linear_learn(wind_list, top_list, pfile) 
    
    #svm_RBF_learn(X, y) 
#    svm_learning(X, y) 
    
    return wind_list, top_list, seq_list, feat_list
#    print(X)

###############################################Input my data for SVM###########################################

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def svm_linear_learn(wind_list, top_list, pfile):
    X = np.array(wind_list)
    y = np.array(top_list)
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

    svc= SVC(kernel='linear', probability=True)
    
##Supervised learning
    svc.fit(X_train, y_train)
    
         
    s = pickle.dumps(svc)
#######################################################Calling the Predictors function #######################################3    
    
    protein_to_predict = open(pfile, 'r+')
    print('Loading the prediction model for your sequence........')
    predictor_svc(protein_to_predict, s)   
     
    
##Supervised Estimators

    y_pred = svc.predict(X_test) #p.random.random(())
    y_pred_prob = svc.predict_proba(X_test)
    feat_feature(y_pred, y_pred_prob, X_test)

 
              
##################################Evaluate my Models Preformance########################

#Accuracy Score
#knn.score(X_test, y_test)
    from sklearn.metrics import accuracy_score
    print('loading the accuracy_score.....')
    print(accuracy_score(y_test, y_pred))

#Classification Report
    from sklearn.metrics import classification_report
    print('loading the classification_report.....')
    print(classification_report(y_test,y_pred))

#ConfusionMatrix
    from sklearn.metrics import confusion_matrix
    print('loading the confusion_matrix.....')
    print(confusion_matrix(y_test, y_pred))

#Cross Validation
    print('loading the cross validation scores')
    score = cross_val_score(svc, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std()*2))
 
 

    return 
   
############################################################################################
   
import pickle  
import numpy as np
# Use of the created predictor model to predict in the whole original dataset
def predictor_svc(protein_to_predict, s):    
    f4pred = protein_to_predict
    #pfile = list(f4pred)
    seq_encode= []
    ids_list = []
    link_list = []
    for line in f4pred:
        #print('this is file line;', line)
        if line[0] == '>':
            line = line.split('\n')
            line = line[0]
            #print(line)
            ids_list.append(line)
        else:
            line = line.split()
            #print(line)
            seq_encode.append(line)   

    #print('This is sequence list', seq_encode)        

    for counter, line in enumerate(seq_encode):
        aa_list = []
        line = line[0]
        #print('This is line ', line)
        for aa in line:
            #print('This is aa', aa)
            i = aadict[aa]
            print(i)
           
            aa_list.append(i)
        link_list.append(aa_list) 
    print(link_list)
    
    pseq_encode = padding(link_list)
    
    X_pred = np.array(pseq_encode)
    print(X_pred.shape)
   
    svc2 = pickle.loads(s)
    y_pred = svc2.predict(X_pred)
    print(y_pred)
    
    top_dict_opp = {0 : 'I', 1 : 'M', 2 : 'O'}
    y_pred = list(y_pred)
    #y_pred_prob = list(y_pred_prob)
    final_list = []
    pre_list = []
    for feat in range(len(X_pred)):
        temp_feat = y_pred[feat]
     #   temp_prob = y_pred_prob[feat]					#Probablities
        #print(temp_feat)
        if temp_feat in top_dict_opp.keys():
            temp_feat = top_dict_opp[temp_feat]
            pre_list.append(temp_feat)
            #final_list.append(temp_prob)
    print('This is the final predicted structure for the protein:')
    print(''.join(pre_list))
    #print(final_list) 
  
    
#############Predicted features##################################   



def feat_feature(y_pred, y_pred_prob, X_test):    
    top_dict_opp = {0 : 'I', 1 : 'M', 2 : 'O'}
    y_pred = list(y_pred)
    #y_pred_prob = list(y_pred_prob)
    final_list = []
    pre_list = []
    for feat in range(len(X_test)):
        temp_feat = y_pred[feat]
        temp_prob = y_pred_prob[feat]					#Probablities
        #print(temp_feat)
        if temp_feat in top_dict_opp.keys():
            temp_feat = top_dict_opp[temp_feat]
            pre_list.append(temp_feat)
            final_list.append(temp_prob)
    print('This is the test structure prediction:')
    print(''.join(pre_list))
#    print(final_list)                        #Probabilities can be printed out but require further processing
            
    return pre_list, final_list #X_test #final_pred



#######################################################################################################




#################################################Confusion Matrix###################################################
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC


def confusion_matrix(wind_list, top_list):
    
    X = np.array(wind_list)
    y = np.array(top_list)
    print(X.shape)
    print(y.shape)
    
    n_samples = len(X) 
    
   
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    
    # We learn the digits on the first half of the digits
    classifier.fit(X[:n_samples / 2], y[:n_samples / 2])
    
    # Now predict the value of the digit on the second half:
    expected = y[n_samples / 2:]
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
    
pfile = input('please enter the filename and extention:')
encoding_file(nfile, pfile)

