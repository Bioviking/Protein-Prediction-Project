



################################################Encoding for SVM function###################################################################
####################################Library imports################################

import numpy as np

#first_dataset = open('../../data/null_dataset/membrane-alpha.3line.txt', 'r+')
nfile = open('../../data/textfile/parsed/both_list.txt', 'r+')
#nfile = open('../../data/textfile/cross_validated/temp_files/test_list70.txt', 'r+')
nfile = input('please enter the filename and extention:')
#encoding_file(nfile, pfile)

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

#def encoding_file(nfile, pfile):
file1 = nfile
wind_list = []
seq_list = []
feat_list = []
top_list = []
link_list = []
window = 0
#Amino acid numbers assignment    

#print(seq_list)
#print(feat_list) 
    
#############################################Encoding the lists##########################3
#seq_list, feat_list = encode_list(file1)
#def encode_list(file1):
seq_list = []
feat_list = []
#ofile = file1
    
for counter, line in enumerate(file1):
   #print(line)
    line = line.strip('\n').split()
#   print(line)
   #print("this is line:", line)
    line = line[0]   
    if counter % 2 == 0:
#      print('This is a Match:', line)
       seq_list.append(line)
    else:
        #print('This is an topology:', line)
        feat_list.append(line)
return seq_list, feat_list
  
    
############################################Loading the binary dictionary###############################3
for counter, line in enumerate(seq_list):
    aa_list = []
    for aa in line:
        i = aadict[aa]
        aa_list.append(i)
    link_list.append(aa_list) 


#######################################################Creating padding###############################################
#def padding(link_list):
    
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
        
        
        




#wind_list = padding(link_list)   # Calling the padding and frame function

##############################Assigning the frames the features
top_list = [top_dict[aa] for pos in feat_list for aa in pos]  

return wind_list, top_list

#END






