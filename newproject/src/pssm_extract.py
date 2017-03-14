import re
import fileinput
import os
import sys

id_file = open('../data/textfile/parsed/id_list.txt', 'r+')
#psi_b_file = open('.../data/fasta_files/psi_blast/', 'r+')
#fname = open('../../data/fasta_files/test_fasta/test_fasta.fasta', 'r+')
#out_test = open('../../data/fasta_files/test_fasta/xfile.fasta', 'w')

pssm_tab = re.compile(" +\d+ [A-Z]")

def parse_fasta(filename):
#    f = open(filename,'r')
    pssm_list = []
    sw = int((wsize - 1)  / 2)
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
                    line = line.split()
                    line = line[22:42]
                    line = [int(i) for i in line]
                    print(line)
                    pssm_list.append(line)
            for num in pssm_list:
                for index in range(len(num)):
                    num[index]= num[index]/100
            
            
            
            for pos in pssm_list:
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
                      
             X.np.array(wind_list)   
             
             index= filename.index(ids)
             new_feat.append(   
                       
        else: 
            print('this is not wortking')
            #pass             
    #print(pssm_list)                
    #sys.exit()
        #       
    return #pssm_list
    
parse_fasta(id_file)
'''           
            
                print(line)
                pssm_tab = re.compile(" +\d+ [A-Z]")
                print(pssm_tab)
                
                    print(line)
                    
                    return
                else:
                    print(line)
'''
        

"""   
                
                
               
                    line = line.strip('\n').split()
                    
                    
        #                    
        
        
    
        
             
            
            
#            print(line)
            
            
    
#    for            
                
"""                
                
'''                
               # 
                #print("this is line:", line)
              #  line = line[0] 
        
               # if counter % 2 == 0:
#            print('This is a Match:', line)
                   # seq_list.append(line)
                
                
   
        
        

        
            
            #print(line)
            temp_id = line
            out_test = open('../../data/fasta_files/fasta_samples/' + temp_id + '.fasta', 'w')
            #print(temp_id)
            out_test.write(line + '\n')
            
        else:
            g_dict[temp_id] = line
            out_test.write(line + '\n')
'''    




 
""" 
def encode_list(file1):
    
    pssm_tab = re.compile(" +\d+ [A-Z]")

    if pssm_tab.match(line):
        print line    
    
    
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
"""  

