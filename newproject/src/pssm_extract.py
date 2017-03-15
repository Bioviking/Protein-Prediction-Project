import re
import fileinput
import os
import sys
import numpy as np

id_feat_file = open('../data/textfile/parsed/id_feat_list.txt', 'r+')
#psi_b_file = open('.../data/fasta_files/psi_blast/', 'r+')
#fname = open('../../data/fasta_files/test_fasta/test_fasta.fasta', 'r+')
#out_test = open('../../data/fasta_files/test_fasta/xfile.fasta', 'w')




top_dict = {'I': 0, 'M': 1, 'O': 2}



pssm_tab = re.compile(" +\d+ [A-Z]")
idslist = []
feat_list = []


def parse_id_feat(id_feat_file):
    fname = id_feat_file
    for counter, line in enumerate(fname):
      line = line.strip()
      if line[0] == '>':
          line = line[0:]
          line = line.split('\n')
          line = line[0]
            #print(line)
          idslist.append(line)
                
          ######Topology###
      else:
          line = line.split('\n')
          line = line[0]
          feat_list.append(line)

        
#        fname.close()       
    return feat_list, idslist



def pssm_parse_encoding(feat_list, idslist, wsize):
#    f = open(filename,'r')
    filename = idslist
    top_file = feat_list
    pssm_list = []
    wsize = wsize
    
    sw = int((wsize - 1) / 2)
    pad =   [0] * 20 #[[0]*20] * sw
    wind_list = []
    new_feat = []
    #g_dict = dict()
    
    
    for ids in filename:
        n_ids = ids.strip()
        print(n_ids)
        if os.path.isfile('../data/fasta_files/psi_blast/%s.fasta.pssm' %n_ids):
            open_file = open('../data/fasta_files/psi_blast/%s.fasta.pssm' %n_ids, 'r+')
            line = open_file.readline()
            
            for counter, line in enumerate(open_file):
                
                pssm_tab = re.compile(" +\d+ [A-Z]")
                #print(pssm_tab)
                
                #print(line)
                if pssm_tab.match(line):                             
                     #.strip('\n')
                    line = line.split()
                    
                    line = line[22:42]
                    #print(line)
                    line = [int(x)/100 for x in line]
                    line = str(line)
                    #print('this is line', line)
                    pssm_list.append(line)
                
            #for num in pssm_list:
             #   for index in range(len(num)):
              #      print('this is index', index)
               #     num[index]= num[index]/100
               #     print('this is num', num[index])
               #     num[index]= str(num[index])
            #print('this is pssm_list', pssm_list)
            
            for pos in pssm_list:
                #print(pos)
                plen = len(pos)
                #print(plen)
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
                    #return wind_list   
            
                   
           
            
            
            index= filename.index(ids)
            new_feat.append(top_file[index])
             
            label_ids = list()
            for feat in new_feat:
               for pos in feat:
                   if pos in top_dict:
                       label_ids.append(top_dict[pos])
                       #return(label_ids)
            
            #return X, y
            #print(y.shape)
        print(wind_list)   
        X=np.array(wind_list) 
        y=np.array(label_ids)
        print(X.shape)
        print(y.shape)               
        #else: 
        #    print('this is not wortking')
            #pass             
    #print(pssm_list)                
    #sys.exit()
        #       
    return X, y#pssm_list



wsize = int(input('Please confirm your window if not default of 3:'))
odd = False
while odd == False:
    if wsize % 2 == 1:
        odd = True
        #sw = int((wsize - 1)  / 2)
        print('Window size is :', wsize)
        parse_id_feat(id_feat_file)
        pssm_parse_encoding(feat_list, idslist, wsize)
    else:
        wsize = int(input('Please enter an odd number or choose default 3:'))





 
        

   





