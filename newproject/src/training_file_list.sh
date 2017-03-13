
################################Creates a .txt of the training files##########################

cd ../data/textfile/cross_validated/
#Making a date folder

for filename in *.txt
do 
    echo $filename >> "./temp_files/file_list.txt"
#    echo hello > testfile01.txt
#    echo hello >> testfile02.txt
done


#for filename in file_list.txt
#do 
#    echo $filename #>> "./temp_files/file_list.txt"
#done
