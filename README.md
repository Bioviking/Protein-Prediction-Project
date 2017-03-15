# Protein_Project - runs with Ubuntu, python 3 and modules of the versions from January 2017 
by Ryno Lawson




Place protein seq to be predicted into the folder:~/Documents/Protein_Project/newproject/src/Predictor/
Read below

Predictor script:
1. predictor.py  - Is a working training file and predictor file. This is hard coded to receive my mainset, train a linear svm model and call in a file to be predicted 
Run the predictor.py script this will take some time as the training of the model and the predictor is combined.



Main dataset:
~/Documents/Protein_Project/newproject/data/null_dataset/membrane-alpha.3line.txt



Script folder:~/Documents/Protein_Project/newproject/src/



Predictor script:
1. predictor.py  - Is a working training file and predictor file. This is hard coded to receive my mainset, train a linear svm model and call in a file to be predicted



Training FolderScripts:


1. parse.py    #Runs parse.py with hard coded dataset file - separates IDs, sequences and features.

2. protein cross.py and cross_valid.py - held my protein level cross validation code.  
#Runs protein_cross function in the cross_valid.py makes 4 training and 1 test dataset file plus extra combo files   

3. training_exe_input.py # Is one of the better scripts for encoding(built on the encoding_file.py- which is where the dynamic windows and encoding was first coded), training svm linear and predicting on. Similiar scripts such as kfold_training_exe.py(tried to loop through varying window size but  and new_rbf.py are modifications of the svm but gear towards rbf or automating my training of different window sizes for the linear_svm(kfold_training)

4.Other python or bash script served as to either create a pipeline between scripts or are templates of code which was write and didnt function. svm_learning.py and svm_functions.py held the sklearn code to be implemented 


#Runs encoding function in encoding.py - training and test files will be looped through here    

In Scripts in Psiblast:

Some of the psiblast scripts I wrote for parsing the fasta files, and running psiblast.

Data folder:

Sorted either by file format(fasta,txt,etc... or stage of processing(cross validating,encoding, etc...)


Result folder:
~/Documents/Protein_Project/newproject/results/

Some training and predictiing results from my svm linear with accuracies and confusion matrix. Some varying window sizes with the kfold files.



Ignore


#| bash training_file_list.sh 
#Runs training_file_list.sh to create a .txt list of file names for training and test function inputs



# This pipeline shell script forms the backbone of the prediction model and runs through the training of the prediction model and quality testing
   


python cross_valid.py #| bash training_file_list.sh  #Runs protein_cross function in the cross_valid.py makes 4 training and 1 test dataset file plus extra combo files   #Runs training_file_list.sh to create a .txt list of file names for training and test function inputs

#bash cross_valid.sh     #Convert dataset to a fasta file for psiblast

python encoding.py
python encoding_linear.py # | python svm_learning.py < (X, y)#(out_sparse1) #, out_formatted    



