


#The daily file setup in data and 
cd data
bash runall.sh 
#mkdate=$(date +'%F')
#cd $mkdate
#mv *.* $mkdate
#cp recombine/proteins.dat ../proteins-saved.dat
cd ..

cd doc
bash runall.sh 
cd ..

cd src
bash runall.sh
cd ..

#Making a date folder

