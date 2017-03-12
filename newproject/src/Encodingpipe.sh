#Created on Fri Mar 10 23:18:32 2017

#@author: ryno


for filename in file_list.txt
do 
    echo $filename #>> "./temp_files/file_list.txt"
done
    python encoding.py encoding($filename) #(out_sparse1) #, out_formatted    

    #Runs encoding function in encoding.py - training and test files will be looped through here    
