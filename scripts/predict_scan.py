# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
from isles22_ensemble import IslesEnsemble
ENSEMBLE_PATH = os.path.dirname(os.getcwd())                                                   # path-to-ensemble-repo
sys.path.append(ENSEMBLE_PATH)

# .nii or DICOM paths
INPUT_FLAIR = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_flair.nii.gz')  # path-to-FLAIR
INPUT_DWI = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_dwi.nii.gz')      # pat-t-DWI
INPUT_ADC = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_adc.nii.gz')      # path-to-ADC
OUTPUT_PATH = os.path.join(ENSEMBLE_PATH, 'example_test')                                      # path-to-output

stroke_segm = IslesEnsemble()
stroke_segm.predict_ensemble(ensemble_path=ENSEMBLE_PATH,
                 input_dwi_path=INPUT_DWI,
                 input_adc_path=INPUT_ADC,
                 input_flair_path=INPUT_FLAIR,
                 output_path=OUTPUT_PATH)
# Options:
#                 fast=True,
#                 save_team_outputs=True,
#                 skull_strip=True)
