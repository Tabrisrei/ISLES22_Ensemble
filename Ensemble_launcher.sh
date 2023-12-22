# Run SEALS Docker (https://github.com/Tabrisrei/ISLES22_SEALS)
# Contact person: Shengbo Gao (GTabris@buaa.edu.cn)
cd SEALS
bash nnunet_launcher.sh

# Run NVAUTO Docker (https://github.com/mahfuzmohammad/isles22)
 # Contact person: Md Mahfuzur Rahman Siddiquee (mrahmans@asu.edu)
cd ../NVAUTO
python process.py

# Run SWAN Docker (https://github.com/pashtari/factorizer-isles22)
# Contact person: Pooya Ashtari (pooya.ashtari@esat.kuleuven.be)
cd ../FACTORIZER
python process.py

cd ..
python majority_voting.py -i output_teams/ -o output/images/stroke-lesion-segmentation/
