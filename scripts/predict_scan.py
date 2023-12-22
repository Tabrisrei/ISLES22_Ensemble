# author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 22.12.2023

import sys
ENSEMBLE_PATH = 'path-to-isles-ensemble-repo/' 
sys.path.append(ENSEMBLE_PATH)
from isles22_ensemble import predict_ensemble


if __name__ == "__main__":

    ''' Example script to run The Isles'22 Ensemble algorithm over a scan.'''

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
