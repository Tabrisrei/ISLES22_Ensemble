# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
from isles22_ensemble import predict_ensemble
ENSEMBLE_PATH = os.path.dirname(os.getcwd())                                                   # path-to-ensemble-repo
sys.path.append(ENSEMBLE_PATH)

# INPUT_FLAIR = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_flair.nii.gz')  # path-to-FLAIR
# INPUT_DWI = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_dwi.nii.gz')      # pat-t-DWI
# INPUT_ADC = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_adc.nii.gz')      # path-to-ADC
# OUTPUT_PATH = os.path.join(ENSEMBLE_PATH, 'example_test')                                      # path-to-output

INPUT_FLAIR = '/home/edelarosa/Documents/datasets/example_dwi/sub-0ab4fbd5/sub-0ab4fbd5_FLAIR_skull-stripped.nii.gz'  # path-to-FLAIR
INPUT_DWI = '/home/edelarosa/Documents/datasets/example_dwi/sub-0ab4fbd5/sub-0ab4fbd5_DWI_space-orig_skull-stripped.nii.gz'      # pat-t-DWI
INPUT_ADC = '/home/edelarosa/Documents/datasets/example_dwi/sub-0ab4fbd5/sub-0ab4fbd5_ADC_space-orig_skull-stripped.nii.gz'      # path-to-ADC
OUTPUT_PATH = '/home/edelarosa/Documents/datasets/example_dwi/sub-0ab4fbd5/test/'                                     # path-to-output

predict_ensemble(isles_ensemble_path=ENSEMBLE_PATH,
                 input_dwi_path=INPUT_DWI,
                 input_adc_path=INPUT_ADC,
                 input_flair_path=INPUT_FLAIR,
                 output_path=OUTPUT_PATH,
                 save_team_outputs=False)
