
#Creating file directory for new programming project

echo "# Protein_Project" >> README.md
git init
git add README.md
mkdir newproject
cd newproject


#Creating files in project directory
touch mkdir.gitkeep
touch README.txt
touch project_plan.docx
touch runall.sh

#Creating of the subdirectories for the new project
mkdir templates
mkdir data
mkdir results
mkdir doc
mkdir src
mkdir bin 

#Creating structure in the data directory

cd data
touch mkdir.gitkeep
touch runall.sh
touch summerize
mkdate=$(date +'%F')
mkdir $mkdate
cd $mkdate
touch mkdir.gitkeep
cd ..
cd ..


#Creating structure in the doc directory

cd doc
touch mkdir.gitkeep
mkdir paper
cd paper
touch mkdir.gitkeep
touch new_paper.docx
cd ..
cd ..



#Creating structure in the results directory

cd results
touch mkdir.gitkeep
touch notebook.html
mkdate=$(date +'%F')
mkdir $mkdate
cd $mkdate
touch mkdir.gitkeep
cd ..
cd ..


#Creating structure in the src directory
cd src
touch mkdir.gitkeep
cd ..



#Creating structure in the bin directory

cd bin
touch mkdir.gitkeep
cd ..

#Returning to the root folder
cd ..

git add .
git commit -m "first commit"
git remote add origin https://github.com/Bioviking/Protein_Project.git
git push -u origin master

