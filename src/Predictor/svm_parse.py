#####################################################3
#Creating a parsed file for the extraction of features for the original dataset.

#library imports


############################################### Load the data from file.



protein_to_predict = input('please enter the filename and extention:')
#encoding_file(nfile, pfile)



out_ids = open('../../data/textfile/parsed/id_list.txt', 'w')
out_ids_seq = open('../../data/textfile/parsed/id_seq_list.fasta', 'w')
out_seq = open('../../data/textfile/parsed/seqlist.txt', 'w')
out_ids_feat = open('../../data/textfile/parsed/id_feat_list.txt', 'w')
out_feat = open('../../data/textfile/parsed/feat_list.txt', 'w')
out_both = open('../../data/textfile/parsed/both_list.txt', 'w')

#next_both = open('../data/textfile/parsed/both_list.txt', 'r+')
######################################################Creating Lists for Ids sequences and features
main_dict = {}
temp_list = []
idslist = []
seqlist = []
feat_list = []

#iterating through the file and separating the id label, sequences and topology

#def predictor_svc(protein_to_predict, s):    
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

return seq_encode







