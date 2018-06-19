#This script looks to conduct sparse encoding on the amino acid sequence



####################################Library imports################################

import numpy as np
import pickle  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from sklearn.svm import SVC
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib as pylab
import numpy as np


#first_dataset = open('../../data/null_dataset/membrane-alpha.3line.txt', 'r+')
nfile = open('../../data/textfile/parsed/both_list.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/trainlist_no1.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/trainlist_no2.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/trainlist_no3.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/trainlist_no4.txt', 'r+')

##################################Creating Lists for Ids sequences and features##############

####Global Variables


# Use of the created predictor model to predict in the whole original dataset
#pfile = input('please enter the filename and extention:')
pfile = '>Q8DIQ1|3kziB.fasta'
#def predictor_svc(protein_to_predict, s):    
protein_to_predict = open(pfile, 'r+')
#print('Loading the prediction model for your sequence........')

f4pred = protein_to_predict
#pfile = list(f4pred)
seq_encode= []
ids_list = []
link_list = []
feat_list = []
seq_list = []

for counter, line in enumerate(nfile):
        #print('this is file line;', line)
    line = line.strip('\n').split()
#        print(line)
    #print("this is line:", line)
    line = line[0]

                
    if counter % 2 == 0:
        #print('This is a Match:', line)
        seq_list.append(line)
    else:
        #print('This is an topology:', line)
        line = line.strip()
        feat_list.append(line)
    #print('This is sequence list', seq_list) 
    #print('This is feature list', feat_list)        
'''
    if line[0] == '>':
        line = line.split('\n')
        line = line[0]
            #print(line)
        ids_list.append(line)
    else:
        line = line.split()
            #print(line)
        seq_encode.append(line)
'''


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
 
for line in seq_list:
    aa_list = []
    #line = line[0]
        #print('This is line ', line)
    for aa in line:
            #print('This is aa', aa)
        i = aadict[aa]
        print(i)  
        aa_list.append(i)
    link_list.append(aa_list) 
print(link_list)
 
###############################################Padding and the window frames##################################  
#pseq_encode = padding(link_list)
pad =   [0] * 20 #[[0]*20] * sw
wind_list= []
top_list= []

wsize = 3
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
            elif aa >= (plen - sw): #                            tempWin = [i for am in pos[(aa-sw):plen] for i in am] + pad*(sw-((plen-1)-aa))
                wind_list.append(tempWin)                         
            else:
                tempWin = [i for am in pos[(aa-sw):(aa+1+sw)] for i in am]
                wind_list.append(tempWin)
            

pseq_encode = wind_list
top_dict = {'I': 0, 'M': 1, 'O': 2}
top_list = [top_dict[aa] for pos in feat_list for aa in pos]   #Assigning the frames the features

########################################Loading into an array and into the predictor#############################

   
X = np.array(pseq_encode)
y = np.array(top_list)
print(X.shape)
print(y.shape)
    
    

# Create a classifier: a support vector classifier
svc = svm.LinearSVC(class_weight='balanced', C=1) 
##Supervised learning
    #X.reshape(-1, 1)
    
score = cross_val_score(svc, X, y, cv=5)
        
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
#print(confusion_matrix(y_test, y_pred))


print(y_test)
print(y_pred)

#n_samples = len(X) 
    
# We learn the digits on the first half of the digits
#svc.fit(X[:n_samples / 2], y[:n_samples / 2])
print("Classification report for classifier %s:\n%s\n"
      % (svc, classification_report(y_test, y_pred)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))






# Plot confusion matrix

classes=[0,1,2]
#clf=svm.LinearSVC(class_weight='balanced')
#clf.fit(X_train,Y_train)
#y_pred = svc.predict(X_test)
cmap=plt.cm.Blues
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Transmembrane - (3 features - window size=3)")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)    

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, round(cm[i, j],5),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()   
savefig('../../results/CM_3fold.png', bbox_inches='tight')  
savefig('../../results/CM_3fold.pdf', bbox_inches='tight')
svc2 = pickle.loads(s)
y_pred = svc2.predict(X_pred)
print(y_pred)


############################################Reverse feature assignment#####################################
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














































'''





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
    #sw = [3, 5, 7, 9, 11, 13]
    wsize = int(input('Please confirm your window if not default of 3:'))
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
    
    confusion_matrix(wind_list, top_list)
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
    
    
    #predictor_svc(protein_to_predict, s)   
     
    
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
    # Plot confusion matrix
    classes=[0,1,2]
    #clf=svm.LinearSVC(class_weight='balanced')
    #clf.fit(X_train,Y_train)
    predicted = clf.predict(X_test)
    cmap=plt.cm.Blues
    cm = confusion_matrix(Y_test, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("TM - 225G (3features -11wsize)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)    

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],5),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#encoding_file(nfile, pfile)
'''

