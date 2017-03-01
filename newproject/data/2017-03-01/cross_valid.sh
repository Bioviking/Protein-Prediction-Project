


#file_len = $(wc -l *.txt > txtlengths.txt)

#for filename in *.fasta
#do
#    echo $filename
#    for i in $file_len
#    do
#        echo $filename 
#        head -n 2 $filename > testing.txt
#        head -n $filename | tail -n 8 > training.txt
#    done
#done

for filename in *.fasta
do
    while read first_line; read second_line
    do
        echo "$first_line" "\n" "$second_line" > 
    done
done
