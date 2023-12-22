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

''' Run the Isles'22 Ensemble algorithm over an entire dataset.
    The data structure follows BIDS convention and the
    open Isles'22 dataset (https://www.nature.com/articles/s41597-022-01875-5)
'''

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

        # print(raw_folder)
        patient_folder_list = sorted(os.listdir(raw_folder))
        for patient_folder in patient_folder_list:
            if patient_folder[:3] != 'sub':
                continue
            raw_patient_folder_path  = os.path.join(raw_folder, patient_folder, 'ses-0001')

            adc_image      = glob(os.path.join(raw_patient_folder_path, '*_adc.nii.gz'))[0]
            dwi_image      = glob(os.path.join(raw_patient_folder_path, '*_dwi.nii.gz'))[0]
            flair_image    = glob(os.path.join(raw_patient_folder_path, '*_flair.nii.gz'))[0]
            
            self.data_dict['adc_image_list'].append(adc_image)
            self.data_dict['dwi_image_list'].append(dwi_image)
            self.data_dict['flair_image_list'].append(flair_image)
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

    # input_path = os.path.join(os.path.dirname(args.data_path), 'input')
    current_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_path, 'input')
    input_image_path = os.path.join(input_path, 'images')
    if not os.path.exists(input_image_path):
        os.makedirs(input_image_path)
    
    
    output_path = os.path.join(current_path, 'output/images/stroke-lesion-segmentation')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.system('chmod 777 -R {}'.format(output_path))
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.system('chmod 777 -R {}'.format(output_path))

    target_path = args.target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    data_path = args.data_path
    dataset_ISLES22 = ISLES22(data_path)
    dataset_ISLES22.load_test_data()

    for index in range(dataset_ISLES22.len):
        dwi_image_path      = dataset_ISLES22.data_dict['dwi_image_list'][index]
        flair_image_path    = dataset_ISLES22.data_dict['flair_image_list'][index]
        adc_image_path      = dataset_ISLES22.data_dict['adc_image_list'][index]

        dwi_image   = sitk.ReadImage(dwi_image_path)
        flair_image = sitk.ReadImage(flair_image_path)
        adc_image   = sitk.ReadImage(adc_image_path)

        sitk_saver(dwi_image, os.path.join(input_image_path, 'dwi-brain-mri', 'dwi_test.nii.gz'))
        sitk_saver(flair_image, os.path.join(input_image_path, 'flair-brain-mri', 'flair_test.nii.gz'))
        sitk_saver(adc_image, os.path.join(input_image_path, 'adc-brain-mri', 'adc_test.nii.gz'))

        os.system('bash docker_test.sh')

        output_file = glob(os.path.join(output_path, '*.nii.gz'))[0]
        output_image = sitk.ReadImage(output_file)

        target_file = dwi_image_path.replace(data_path, target_path).split('_dwi.nii.gz')[0] + '_seg.nii.gz'
        sitk_saver(output_image, target_file)

    shutil.rmtree(input_path)
    shutil.rmtree(output_path)
        

