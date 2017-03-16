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
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
#first_dataset = open('../../data/null_dataset/membrane-alpha.3line.txt', 'r+')
#nfile = open('../../data/textfile/parsed/both_list.txt', 'r+')
nfile = open('../../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')
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
    
X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1  
'''
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
'''
##############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

##############################################################################
# visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
