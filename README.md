![alt text](https://github.com/Tabrisrei/ISLES22_Ensemble/blob/master/ensemble-logo.png)

# The ISLES'22 Ensemble Algorithm 
## Introduction
Algorithm to predict ischemic stroke lesions from MRI data.

This repository contains an ensemble algorithm devised in the context of the ISLES'22 MICCAI Challenge (https://isles22.grand-challenge.org/).
The top-3 leading algorithms are used in a majority-voting scheme to predict outputs.


## 1) Installation
1.1) In your conda environment 'myenv' install basic pytorch, torchvision and cudatoolkit.

**Note: Mandatory Python version 3.8.0**

```bash
conda activate myenv
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

1.2) Clone this repository.

```bash
cd ~
git clone https://github.com/Tabrisrei/ISLES22_Ensemble.git
```

1.3) Install dependencies for the ensemble algorithm .

```bash
cd ISLES22_Ensemble
pip install -e SEALS/
pip install -e FACTORIZER/model/factorizer/
pip install -r requirements.txt
```

If successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

1.4) Download all content from https://drive.google.com/drive/folders/1GQP5nmtoIfhyu2LHCzPVYnZZ7ZQDX1go?usp=drive_link

1.5) 
- Decompress the file "nnUNet_trained_models.tar.gz" in `~/ISLES22_Ensemble/SEALS/data/`.
- Decompress the file "ts.tar.gz" in `~/ISLES22_Ensemble/NVAUTO/`.
- Decompress "log.tar.gz" in `~/ISLES22_Ensemble/FACTORIZER/model/`.


## 2) Usage
The algorithms works over **skull-stripped MRI images**, directly in native-space (no image co-registration is needed). Image modalities required for running the algorithm are DWI (b=1000), ADC and FLAIR. 

**Note: DWI and ADC images are supposed to be obtained from the same acquisition timepoint. Differences between them (e.g. affine, voxel resolution, number of voxels, etc.) might impact the algorithm performance.**

2.1) From Python

```bash
import sys
ENSEMBLE_PATH = 'path-to-isles-ensemble-repo' 
sys.path.append(ENSEMBLE_PATH)
from isles22_ensemble import predict_ensemble

INPUT_FLAIR = 'path-to-flair.nii.gz'
INPUT_ADC = 'path-to-adc.nii.gz'
INPUT_DWI = 'path-to-dwi.nii.gz'
OUTPUT_PATH = 'path-to-output-folder'

predict_ensemble(isles_ensemble_path=ENSEMBLE_PATH,
                     input_dwi_path=INPUT_DWI,
                    input_adc_path=INPUT_ADC,
                    input_flair_path=INPUT_FLAIR,
                    output_path=OUTPUT_PATH,
                    save_team_outputs=False)
```

2.2) From terminal

Place your MRI images (.nii.gz) in
```bash
~/ISLES22_Ensemble/input/images/adc-brain-mri/
~/ISLES22_Ensemble/input/images/dwi-brain-mri/
~/ISLES22_Ensemble/input/images/flair-brain-mri/
```

Run: 
```bash
bash Ensemble_launcher.sh
```

Results will be stored in `~/ISLES22_Ensemble/output/images/stroke-lesion-segmentation/`. 

## Citation
Please cite the following manuscripts when using The Isles'22 Ensemble

* de la Rosa, E., Reyes, M., Liew, S. L., Hutton, A., Wiest, R., Kaesmacher, J., ... & Wiestler, B. (2024). A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge. arXiv preprint arXiv:2403.19425.

* Hernandez Petzsche, M. R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela, W., Reyes, M., ... & Kirschke, J. S. (2022). ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. Scientific data, 9(1), 762.


## About the Ensembled Algorithms 
* Algorithm SEALS is based on nnUnet. Git repo https://github.com/Tabrisrei/ISLES22_SEALS 

* Algorithm NVAUTO is based on MONAI Auto3dseg. Git repo: https://github.com/mahfuzmohammad/isles22

* Algorithm SWAN is based on FACTORIZER. Git repo: https://github.com/pashtari/factorizer-isles22


## Questions
Please contact Ezequiel de la Rosa (ezequiel.delarosa@uzh.ch) or Shengbo Gao (gtabris@buaa.edu.cn).

## Acknowledgement
We thank all ISLES'22 challenge participants, collaborators and organizers for allowing this work to happen. We also thank all developers and maintaners of the repositories named below for sharing such valuable resources.
- The code of Team SEALS is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 
- The code of NVAUTO is adapted from [MONAI](https://github.com/Project-MONAI/MONAI)
- The code of SWAN is adapted from [FACTORIZER](https://github.com/pashtari/factorizer)
