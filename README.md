# MICCAI Ischemic Stroke LEsion Segmentation 2022 Top 3 Major Voting Ensemble

## Introduction

This is the repository of Top3 team SEALS/NVAUTO/SWAN final algorithm major voting for MICCAI ISLES22 competition. If you are interested in our work, please cite 《XXX》

## Usage

This repository is based on nnUNet/MONAI/FACTORIZER. Please follow the installation instruction below. We recommend to install required packages under conda environment.

## Installation

If you do not want to install conda, please skip to the step 3.

1. Let us hypothesis you are using conda environment named "isles", activate conda environment "isles"

```bash
conda activate isles
```

2. Run the following command to install basic pytorch、torchvision library and corresponding cudatoolkit.

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Clone this repository.

```bash
cd ~
git clone https://github.com/Tabrisrei/ISLES22_Ensemble.git
```

4. Run the following command to install nnUNet package requirements of SEALS.

```bash
cd ISLES22_Ensemble
pip install -e SEALS/
```

5. Run the following command to install nnUNet package requirements of SEALS.

```bash
cd ISLES22_Ensemble
pip install -e FACTORIZER/model/factorizer/
```

6. Run the following command to install MONAI package requrements of NVAUTO. It also helps check if successfully installed required package version. (nnunet installation process did not strictly determine the package version, which may raise some unpredictable error.)

```bash
cd ISLES22_SEALS
pip install -r requirements.txt
```

If successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

5. Download zip files of top3 models from [Google Drive](https://drive.google.com/drive/folders/1GQP5nmtoIfhyu2LHCzPVYnZZ7ZQDX1go?usp=drive_link), there are 3 compressed files in it: "nnUNet_trained_models.tar.gz", "ts.tar.gz" and "log.tar.gz".
6. Decompression the "nnUNet_trained_models.tar.gz" file and put the "nnUNet_trained_models" folder you've got into the directory `~/ISLES22_Ensemble/SEALS/data/`.
7. Decompression the "ts.tar.gz" file and put the "ts" folder you've got into the directory `~/ISLES22_Ensemble/NVAUTO/`.
8. Decompression the "log.tar.gz" file and put the "log" folder you've got into the directory `~/ISLES22_Ensemble/FACTORIZER/model/`.

Now we have prepared all the required code and models, you can follow the steps below to test your own dataset. 

9. Convert your dwi and adc image to mha format (Do not forget metadata)
10. Put your dwi and adc data into `/input/images/dwi-brain-mri/` and `/input/images/adc-brain-mri/` folder respectively.
11. Run the following command to generate the results.

```bash
bash Ensemble_launcher.sh
```

After the results are generated, you can check the results in `/output/images/stroke-lesion-segmentation/` folder.

## Questions

Please contact gtabris@buaa.edu.cn or ezequiel.delarosa@icometrix.com

## Acknowledgement

- The code of Team SEALS is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), We thank Dr. Fabian Isensee etc. for their elegant and efficient code base.
- The code of NVAUTO is adapted from [MONAI](https://github.com/Project-MONAI/MONAI)
- The code of SWAN is adapted from [FACTORIZER](https://github.com/pashtari/factorizer)

Thanks for your interest in our work. If you have any questions, please feel free to contact us.