# run seals
cd SEALS
bash nnunet_launcher.sh

# run nvauto
cd ../NVAUTO
python process.py

# run factorizer
cd ../FACTORIZER
python process.py

cd ..
python major_voting.py -i /output_teams/ -o /output/images/stroke-lesion-segmentation/