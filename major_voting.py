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
        flair_folder  = 'flair-brain-mri'

        self.dwi_path   = glob(os.path.join(self.root, dwi_folder, '*.mha'))[0]
        self.adc_path   = glob(os.path.join(self.root, adc_folder, '*.mha'))[0]
        self.flair_path = glob(os.path.join(self.root, flair_folder, '*.mha'))[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_folder", required=True, help="folders contain ")
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving evaluation csv")
    args = parser.parse_args()

    input_folder  = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load origin mha file path
    raw_data_dir = '/input/images'
    dataset_ISLES22 = ISLES22(raw_data_dir)
    dataset_ISLES22.load_data()

    # load mha image
    image_file = sitk.ReadImage(dataset_ISLES22.dwi_path)

    # major voting

    result_array = None
    sub_folders = os.listdir(input_folder)
    for folder in sub_folders:
        try:
            pred_file = glob(os.path.join(input_folder, folder, 'images', 'stroke-lesion-segmentation', '*.mha'))[0]
            pred_image = sitk.ReadImage(pred_file)
            pred_array = sitk.GetArrayFromImage(pred_image)
        except:
            print('no prediction! generating full 0 mask!')
            image_array = sitk.GetArrayFromImage(image_file)
            pred_array  = np.zeros_like(image_array)


        if result_array is None:
            result_array = np.zeros_like(pred_array)
        else:
            result_array += pred_array

    result_array = np.where(result_array>2, 1, 0)
    result_image = sitk.GetImageFromArray(result_array)

        
    result_image.SetOrigin(image_file.GetOrigin())
    result_image.SetSpacing(image_file.GetSpacing())
    result_image.SetDirection(image_file.GetDirection())
    sitk.WriteImage(result_image, os.path.join(output_folder, dataset_ISLES22.dwi_path.split('/')[-1]))

    # dump the result to json file
    case_results = []
    json_result =   {"outputs": [dict(
                                    type="Image", 
                                    slug="stroke-lesion-segmentation",
                                    filename=str(dataset_ISLES22.dwi_path.split('/')[-1]))],
                    "inputs": [dict(
                                    type="Image", 
                                    slug="dwi-brain-mri",
                                    filename=str(dataset_ISLES22.dwi_path.split('/')[-1]))]}
    case_results.append(json_result)
    json_writer(os.path.join(output_folder, 'result.json'), case_results)
