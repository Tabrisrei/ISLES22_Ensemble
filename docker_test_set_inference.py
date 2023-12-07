import os
import pdb
import json
import shutil
import argparse
from glob import glob
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy

import numpy as np
import nibabel as nib
import SimpleITK as sitk


class ISLES22():
    def __init__(self, root='data_ISLES22/dataset-ISLES22/'):
        self.root = root
        self.data_dict = {}

    def load_test_data(self):
        raw_folder    = 'rawdata'
        raw_folder    = os.path.join(self.root, raw_folder)
        
        self.data_dict['adc_image_list']   = []
        self.data_dict['dwi_image_list']   = []
        self.data_dict['flair_image_list'] = []
        self.data_dict['adc_json_list']    = []
        self.data_dict['dwi_json_list']    = []
        self.data_dict['flair_json_list']  = []

        patient_folder_list = sorted(os.listdir(raw_folder))
        for patient_folder in patient_folder_list:
            if patient_folder[:3] != 'sub':
                continue
            raw_patient_folder_path  = os.path.join(raw_folder, patient_folder, 'ses-0001')

            adc_image      = glob(os.path.join(raw_patient_folder_path, '*_adc.nii.gz'))[0]
            dwi_image      = glob(os.path.join(raw_patient_folder_path, '*_dwi.nii.gz'))[0]
            flair_image    = glob(os.path.join(raw_patient_folder_path, '*_flair.nii.gz'))[0]
            try:
                adc_json       = glob(os.path.join(raw_patient_folder_path, '*_adc.json'))[0]
                dwi_json       = glob(os.path.join(raw_patient_folder_path, '*_dwi.json'))[0]
                flair_json     = glob(os.path.join(raw_patient_folder_path, '*_flair*.json'))[0]
            except:
                adc_json = None
                dwi_json = None
                flair_json = None
            
            self.data_dict['adc_image_list'].append(adc_image)
            self.data_dict['dwi_image_list'].append(dwi_image)
            self.data_dict['flair_image_list'].append(flair_image)
            self.data_dict['adc_json_list'].append(adc_json)
            self.data_dict['dwi_json_list'].append(dwi_json)
            self.data_dict['flair_json_list'].append(flair_json)
        self.len = len(self.data_dict['dwi_image_list'])

def sitk_saver(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    sitk.WriteImage(image, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ISLES22 Docker Test Set Inference')
    parser.add_argument('--target_path', 
                        type=str, 
                        default='~/test_results/', 
                        help='final result root path, pick a place you like')
    parser.add_argument('--data_path', 
                        type=str, 
                        default='data_ISLES22/dataset-ISLES22/', 
                        help='dataset root path, should be the parent directory of rawdata')
    args = parser.parse_args()

    input_path = '~/input/'
    input_image_path = os.path.join(input_path, 'images')
    if not os.path.exists(input_image_path):
        os.makedirs(input_image_path)

    target_path = args.target_path

    data_path = args.data_path
    dataset_ISLES22 = ISLES22(data_path)
    dataset_ISLES22.load_test_data()

    for index in range(dataset_ISLES22.len):
        dwi_image_path      = dataset_ISLES22.data_dict['dwi_image_list'][index]
        flair_image_path    = dataset_ISLES22.data_dict['flair_image_list'][index]
        adc_image_path      = dataset_ISLES22.data_dict['adc_image_list'][index]
        dwi_json_path       = dataset_ISLES22.data_dict['dwi_json_list'][index]
        flair_json_path     = dataset_ISLES22.data_dict['flair_json_list'][index]
        adc_json_path       = dataset_ISLES22.data_dict['adc_json_list'][index]

        dwi_image   = sitk.ReadImage(dwi_image_path)
        flair_image = sitk.ReadImage(flair_image_path)
        adc_image   = sitk.ReadImage(adc_image_path)

        sitk_saver(dwi_image, os.path.join(input_image_path, 'dwi_brain_mri', 'dwi_test.mha'))
        sitk_saver(flair_image, os.path.join(input_image_path, 'flair_brain_mri', 'flair_test.mha'))
        sitk_saver(adc_image, os.path.join(input_image_path, 'adc_brain_mri', 'adc_test.mha'))

        shutil.copy(dwi_json_path, os.path.join(input_path, 'dwi_mri_acquisition_parameters.json'))
        shutil.copy(flair_json_path, os.path.join(input_path, 'flair_mri_acquisition_parameters.json'))
        shutil.copy(adc_json_path, os.path.join(input_path, 'adc_mri_parameters.json'))

        os.system('docker run --rm \
                  --memory="40g" \
                  --memory-swap="40g" \
                  --network="none" \
                  --cap-drop="ALL" \
                  --security-opt="no-new-privileges" \
                  --shm-size="128m" \
                  --pids-limit="256" \
                  -v ~/input/:/input/ \
                  -v ~/output/:/output/ \
                  gtabris/isles_major_voting:latest')
        
        try:
            output_path = glob('~/output/images/stroke-lesion-segmentation/*')[0]
            output_image = sitk.ReadImage(output_path)
            sitk_saver(output_image, 
                       os.path.join(target_path, 
                                    dwi_image_path.split(data_path))[-1].split('_dwi.nii.gz')[0] + '_seg.nii.gz')
        except:
            continue

        

