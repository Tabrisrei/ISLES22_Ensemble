# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
import time
import glob
import numpy as np
import pandas as pd

ENSEMBLE_PATH = os.path.dirname(os.getcwd())  # path-to-ensemble-repo
#sys.path.append(ENSEMBLE_PATH)
#sys.path.append(os.path.join(ENSEMBLE_PATH, 'src'))
from src.isles22_ensemble import IslesEnsemble

# .nii or DICOM folders
# INPUT_FLAIR = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_flair.nii.gz')  # path-to-FLAIR
# INPUT_DWI = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_dwi.nii.gz')      # pat-t-DWI
# INPUT_ADC = os.path.join(ENSEMBLE_PATH, 'data', 'sub-strokecase0001_ses-0001_adc.nii.gz')      # path-to-ADC
# OUTPUT_PATH = os.path.join(ENSEMBLE_PATH, 'example_test')                                      # path-to-output
#all_times = []
out_folder = '/mnt/hdda/edelarosa/ensemble_results_jh'
# out_folder = '/media/data/stroke/ensemble_24_results/'

for input_case in glob.glob('/mnt/hdda/edelarosa/jh_example/dataset*/raw_data/*'):
    OUTPUT_PATH = os.path.join(out_folder, input_case.split('/')[-1])
    if not os.path.exists(OUTPUT_PATH):

        clinical_data = pd.read_table(input_case.replace('raw_data', 'phenotype') + '_demographic_clinical.tsv')
        if clinical_data['Lesion-type'][0] == 'ischemic':  # only ischemic lesions

            os.mkdir(OUTPUT_PATH)
            print('running:', input_case)
            # new inputs to dcm
            INPUT_FLAIR = glob.glob(os.path.join(input_case, 'anat', '*FLAIR.nii.gz'))
            INPUT_DWI = glob.glob(os.path.join(input_case, 'DWI', '*DWI.nii.gz'))
            INPUT_ADC = glob.glob(os.path.join(input_case, 'DWI', '*ADC.nii.gz'))
            if len(INPUT_FLAIR) > 0 and len(INPUT_DWI) > 0 and len(INPUT_ADC) > 0:  # skip scans missing flair
                INPUT_FLAIR = INPUT_FLAIR[0]
                INPUT_DWI = INPUT_DWI[0]
                INPUT_ADC = INPUT_ADC[0]
                #OUTPUT_PATH = '/home/edelarosa/Documents/datasets/example_dwi/test_me'

                stroke_segm = IslesEnsemble()

                # Start timing
                start_time = time.time()

                # Run the ensemble prediction
                stroke_segm.predict_ensemble(ensemble_path=ENSEMBLE_PATH,
                                             input_dwi_path=INPUT_DWI,
                                             input_adc_path=INPUT_ADC,
                                             input_flair_path=INPUT_FLAIR,
                                             output_path=OUTPUT_PATH,
                                             fast=True,
                                             save_team_outputs=False,
                                             skull_strip=True,
                                             results_mni=True)

        # End timing
#         end_time = time.time()
#
#         # Calculate and print the elapsed time in minutes
#         elapsed_time_minutes = (end_time - start_time) / 60
#         all_times.append(elapsed_time_minutes)
#
# print(np.mean(np.asarray(all_times)))
# print(np.std(np.asarray(all_times)))
# print(np.min(np.asarray(all_times)))
# print(np.max(np.asarray(all_times)))

# print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")
