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
nfile = open('../../../data/textfile/parsed/both_list.txt', 'r+')
#nfile = open('../../../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')
#nfile = open('../../../data/textfile/cross_validated/trainlist_no1.txt', 'r+')
#nfile = open('../../../data/textfile/cross_validated/trainlist_no2.txt', 'r+')
#nfile = open('../../../data/textfile/cross_validated/trainlist_no3.txt', 'r+')
#nfile = open('../../../data/textfile/cross_validated/trainlist_no4.txt', 'r+')

##################################Creating Lists for Ids sequences and features##############

####Global Variables


# Use of the created predictor model to predict in the whole original dataset
#pfile = input('please enter the filename and extention:')
#pfile = '../>Q8DIQ1|3kziB.fasta'
#def predictor_svc(protein_to_predict, s):    
#protein_to_predict = open(pfile, 'r+')
#print('Loading the prediction model for your sequence........')
#pfile = open('../>Q8DIQ1|3kziB.fasta', 'r+') 
#f4pred = protein_to_predict
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

wsize = 31
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
#svc = svm.SVC(kernel='rbf', class_weight='balanced', C=1) 
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
savefig('../../../results/CM_31fold.png', bbox_inches='tight')  
savefig('../../../results/CM_31fold.pdf', bbox_inches='tight')
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

