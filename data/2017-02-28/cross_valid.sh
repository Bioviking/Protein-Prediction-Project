


$ wc -l *.pdb

for filename in *.fasta
do
    echo $filename 
    head -n 10 $filename | tail -n 8 > training.txt
done
