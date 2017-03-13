#Created on Thu Mar  9 21:23:08 2017

#@author: ryno


################################Pipeline######################################################

# This pipeline shell script forms the backbone of the prediction model and runs through the training of the prediction model and quality testing
   
python parse.py    #Runs parse.py with hard coded dataset file

python protein_cross.py #| bash training_file_list.sh  #Runs protein_cross function in the cross_valid.py makes 4 training and 1 test dataset file plus extra combo files   #Runs training_file_list.sh to create a .txt list of file names for training and test function inputs

#bash cross_valid.sh     #Convert dataset to a fasta file for psiblast

#python shuffle.py    #Run parse.py with hard coded dataset file


python encoding.py
python encoding_linear.py # | python svm_learning.py < (X, y)#(out_sparse1) #, out_formatted    
#Runs encoding function in encoding.py - training and test files will be looped through here    



