import os
import sys
import json
import shutil
import argparse
import numpy as np
from glob import glob
import SimpleITK as sitk

def json_writer(json_path, data):
    with open(str(json_path), "w") as f:
        json.dump(data, f)

def sitk_loader(image_path):
    image = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(image)
    return image, array

def stik_saver(origin_image, array, image_path):
    # set the image meta data
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin_image.GetOrigin())
    image.SetSpacing(origin_image.GetSpacing())
    image.SetDirection(origin_image.GetDirection())
    # save the image
    sitk.WriteImage(image, image_path)


class ISLES22():
    def __init__(self, root):
        self.root  = root
        self.data_dict = {}

    def load_data(self):
        dwi_folder    = 'dwi-brain-mri'
        adc_folder    = 'adc-brain-mri'
        #flair_folder  = 'flair-brain-mri'

        self.dwi_path   = glob(os.path.join(self.root, dwi_folder, '*.nii.gz'))[0]
        self.adc_path   = glob(os.path.join(self.root, adc_folder, '*.nii.gz'))[0]
        #self.flair_path = glob(os.path.join(self.root, flair_folder, '*.nii.gz'))[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_folder", required=True, help="folders contain ")
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving evaluation csv")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load origin file path
    raw_data_dir = 'input/images'
    dataset_ISLES22 = ISLES22(raw_data_dir)
    dataset_ISLES22.load_data()

    # load image
    image_file = sitk.ReadImage(dataset_ISLES22.dwi_path)

    # majority voting
    result_array = None
    #sub_folders = os.listdir(input_folder)
    teams = ['seals', 'nvauto', 'factorizer']
    pred_array = {}
    for folder in teams :
        try:
            pred_file = glob(os.path.join(input_folder, folder, '*.nii.gz'))[0]
            pred_image = sitk.ReadImage(pred_file)
            pred_array[folder] = sitk.GetArrayFromImage(pred_image).astype(np.int8)
        except:
            0
    # majority voting - all outputs available
    if all(key in pred_array for key in teams):
        print('Running Majority voting ...')
        print()
        result_array = pred_array['seals'] + pred_array['nvauto'] + pred_array['factorizer']
        result_array = result_array/3 > 0.5
    # backup- if one algorithm fails, return results from the available best-ranked team.
    elif 'seals' in pred_array.keys():
        result_array = pred_array['seals']
        print('Returning results from SEALS `algorithm`.')
        print()
    elif 'nvauto' in pred_array.keys():
        result_array = pred_array['nvauto']
        print('Returning results from NVAUTO algorithm.')
        print()
    else:
        result_array = pred_array['factorizer']
        print('Returning results from FACTORIZER algorithm.')
        print()

    # Write results
    result_image = sitk.GetImageFromArray(result_array.astype(np.uint8))
    result_image.SetOrigin(image_file.GetOrigin())
    result_image.SetSpacing(image_file.GetSpacing())
    result_image.SetDirection(image_file.GetDirection())
    sitk.WriteImage(result_image, os.path.join(output_folder, 'lesion_msk.nii.gz'))

