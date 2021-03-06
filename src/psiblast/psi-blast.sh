# run a psi-blast search 


#change directory to uniref90 or uniref50 
export BLASTDB=/local_uniref/uniref/uniref90

#cd ~/Documents/Protein_Project/newproject/data/fasta_files/fasta_samples/


for seq in *.fasta ; do #specify the folder where all FASTA files with single sequences are stored


#Now, the if statement is sort of a "safety step" to check if the files have been created already in the output directory in case that the computer shuts down or others


if [ ! -f $seq.psiblast ] ; then
	echo "Running psiblast on $seq at $(date)..."
	time psiblast -query $seq -db uniref90.db -num_iterations 3 -evalue 0.001 -out $seq.psiblast -out_ascii_pssm $seq.pssm -num_threads 8
# The line above runs psiblast for each sequence. You have to speficy the number of iterations, the evalue and -num_threads (number of cores employed to run the program. It depends on the computer/laptop).

	echo "Finished running psiblast on $seq at $(date)."
fi
done

#Know when the iterations are done we will echo:
echo 'PSI-BLAST run is complete'

# Move all the files from the data_files folder to a psiblast_output folder (as desired)
cd ~/Documents/Protein_Project/newproject/data/fasta_files/fasta_samples/

mv *.psiblast ~/Documents/Protein_Project/newproject/data/fasta_files/psi_blast/
mv *.pssm ~/Documents/Protein_Project/newproject/data/fasta_files/psi_blast/
