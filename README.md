![alt text](https://github.com/Tabrisrei/ISLES22_Ensemble/blob/master/ensemble-logo.png)

# The ISLES'22 Ensemble Algorithm 
## Introduction
Algorithm to predict ischemic stroke lesions from MRI data.

This repository contains an ensemble algorithm devised in the context of the ISLES'22 MICCAI Challenge (https://isles22.grand-challenge.org/).
The top-3 leading algorithms are used in a majority-voting scheme to predict outputs.

1. [Installation](#installation)
2. [Usage](#usage)
3. [Get started](#get-started)
4. [Citation](#citation)
5. [About the Ensembled Algorithms](#about-the-ensembled-algorithms)
6. [Questions](#questions)
7. [Acknowledgement](#acknowledgement)


## Installation
1.1) Clone this repository.

```bash
git clone https://github.com/Tabrisrei/ISLES22_Ensemble.git
cd ISLES22_Ensemble
```

1.2) Create a conda environment and install dependencies. 
**Note: Mandatory Python version 3.8.0 (!)**

```bash
conda create --name isles_ensemble python=3.8.0 pip=23.3.1
conda activate isles_ensemble
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge openslide-python
conda install python=3.8.0 # important, since pytorch triggers the installation of later python versions
pip install -e SEALS/
pip install -e FACTORIZER/model/factorizer/
pip install -r requirements.txt
```

If successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

1.3) Download the model weights from [here](https://drive.google.com/drive/folders/1_NqCVS88coFdkzYPzOapVKlhmdqKOhqK?zx=67gokxezdc8f) and decompress the file.
Your directory shoul look like:
```

ISLES22_Ensemble/
├── weights/
│   ├── SEALS/
│   │   └── (...)
│   ├── NVAUTO/
│   │   └── (...)
│   └── FACTORIZER/
│       └── (...)
```


## Usage
The algorithms works over **skull-stripped MRI images**, directly in native-space (no image co-registration is needed). Image modalities required for running the algorithm are DWI (b=1000), ADC and FLAIR. 

**Note: DWI and ADC images are supposed to be obtained from the same acquisition timepoint. Differences between them (e.g. affine, voxel resolution, number of voxels, etc.) might impact the algorithm performance.**

From Python

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

From terminal

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

## Get started 
You can try out our algorithm using a real test scan located in `~/data`. For doing so, run the example script located in `~/scripts/predict_scan.py`.
The example data belongs to the ISLES'22 dataset (Hernandez Petzsche et al., Sci Data 2022).

## Citation
Please cite the following manuscripts when using The Isles'22 Ensemble:

* de la Rosa, E., Reyes, M., Liew, S. L., Hutton, A., Wiest, R., Kaesmacher, J., ... & Wiestler, B. (2024). A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge. arXiv preprint arXiv:2403.19425.

* Hernandez Petzsche, M. R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela, W., Reyes, M., ... & Kirschke, J. S. (2022). ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. Scientific data, 9(1), 762.


## About the Ensembled Algorithms 
* Algorithm SEALS is based on nnUnet. Git [repo](https://github.com/Tabrisrei/ISLES22_SEALS) 

* Algorithm NVAUTO is based on MONAI Auto3dseg. Git [repo](https://github.com/mahfuzmohammad/isles22)

* Algorithm SWAN is based on FACTORIZER. Git [repo](https://github.com/pashtari/factorizer-isles22)


## Questions
Please contact Ezequiel de la Rosa (ezequiel.delarosa@uzh.ch) or Shengbo Gao (gtabris@buaa.edu.cn).

## Acknowledgement
We thank all ISLES'22 challenge participants, collaborators and organizers for allowing this work to happen. We also thank all developers and maintaners of the repositories named below for sharing such valuable resources.
- The code of Team SEALS is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 
- The code of NVAUTO is adapted from [MONAI](https://github.com/Project-MONAI/MONAI)
- The code of SWAN is adapted from [FACTORIZER](https://github.com/pashtari/factorizer)
