
############################Files##########################################3
out_file = open('../../result/classification_report.txt', 'w')
 
##################SVM_Linear_Model_Trainer_ and _Test#################################################


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

#def svm_linear_learn(wind_list, top_list, pfile):


X = np.array(wind_list)
y = np.array(top_list)
print(X.shape)
print(y.shape)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

###################################Creating my Model##############################3
##Supervised Learning Estimators

svc= SVC(kernel='linear', probability=True)
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=5, random_state=0, max_features='auto')    
rbf= SVC(kernel='rbf')    

######################################SVM_RBF_Supervised learning############################
 
rbf = rbf.fit(X_train, Y_train)
print(rbf.score(X_test,Y_test)) 

#################################RandomForest_Supervised learning#####################

clf = clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))    
    

#################################SVM_linear_Supervised learning############
svc.fit(X_train, y_train)
s = pickle.dumps(svc) 
    


y_pred = svc.predict(X_test) #p.random.random(())
y_pred_prob = svc.predict_proba(X_test)
#feat_feature(y_pred, y_pred_prob, X_test)
 
              
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
out_file.write(classification_report(y_test,y_pred))

#ConfusionMatrix
from sklearn.metrics import confusion_matrix
print('loading the confusion_matrix.....')
print(confusion_matrix(y_test, y_pred))

#Cross Validation
print('loading the cross validation scores')
score = cross_val_score(svc, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std()*2))


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

 
