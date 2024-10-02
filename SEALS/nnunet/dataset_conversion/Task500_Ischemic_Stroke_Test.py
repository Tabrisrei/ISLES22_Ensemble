#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pdb
import shutil
from glob import glob
from pathlib import Path

import string
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict
from multiprocessing.pool import Pool

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.stats as ss
from skimage import transform
from medpy.metric import dc, hd95
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

class ISLES22():
    def __init__(self, root):
        self.root = root
        self.data_dict = {}

    def load_data(self):
        dwi_folder = 'dwi'
        adc_folder = 'adc'
        # flair_folder  = 'flair-brain-mri'

        # self.dwi_path = glob(os.path.join(self.root, dwi_folder, '*.nii.gz'))[0]
        # self.adc_path = glob(os.path.join(self.root, adc_folder, '*.nii.gz'))[0]
        # self.flair_path = glob(os.path.join(self.root, flair_folder, '*.nii.gz'))[0]

        dwi_path = os.path.join(self.root, dwi_folder, dwi_folder + '.nii.gz')
        ss_dwi_path = os.path.join(self.root, dwi_folder, dwi_folder + '_ss.nii.gz')
        if os.path.exists(ss_dwi_path):
            self.dwi_path = ss_dwi_path
        elif os.path.exists(dwi_path):
            self.dwi_path = dwi_path

        adc_path = os.path.join(self.root, adc_folder, adc_folder + '.nii.gz')
        ss_adc_path = os.path.join(self.root, adc_folder, adc_folder + '_ss.nii.gz')

        if os.path.exists(ss_adc_path):
            self.adc_path = ss_adc_path
        elif os.path.exists(adc_path):
            self.adc_path = adc_path
#        self.flair_path = glob(os.path.join(self.root, flair_folder, '*.nii.gz'))[0]


def respacing_file(image_file, target_spacing, resample_method):
    """
    Respacing file to target space size
    :param image_file: sitk.SimpleITK.Image
    :param target_spacing: np.array([H_space, W_space, D_space])
    :resample_method: SimpleITK resample method (e.g. SimpleITK.sitkLinear, SimpleITK.sitkNearestNeighbor)
    :return: resampled_image_file: sitk.SimpleITK.Image
    """
    if type(image_file) is not sitk.SimpleITK.Image:
        image_file = sitk.ReadImage(image_file)
    if not isinstance(target_spacing, np.ndarray):
        target_spacing = np.array(target_spacing)

    # initialize resampler
    resampler_image = sitk.ResampleImageFilter()

    # set target size
    origin_direction = np.array(image_file.GetDirection())
    origin_spacing   = np.array(image_file.GetSpacing())
    origin_size      = np.array(image_file.GetSize())
    factor           = np.array(target_spacing / origin_spacing)
    target_size      = (origin_size / factor).astype(np.uint8)

    # set the parameters of image
    resampler_image.SetReferenceImage(image_file)  # set rasampled image meta data same to origin data
    resampler_image.SetOutputSpacing(target_spacing.tolist())  # set target image space
    resampler_image.SetSize(target_size.tolist())  # set target image size
    resampler_image.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler_image.SetInterpolator(resample_method)

    # launch the resampler
    resampled_image_file = resampler_image.Execute(image_file)

    return resampled_image_file


def reimplement_resize(image_file, target_file, resample_method=sitk.sitkLinear):
    """
    Respacing file to target space size
    :param image_file: sitk.SimpleITK.Image
    :param target_spacing: np.array([H_space, W_space, D_space])
    :resample_method: SimpleITK resample method (e.g. SimpleITK.sitkLinear, SimpleITK.sitkNearestNeighbor)
    :return: resampled_image_file: sitk.SimpleITK.Image
    """
    # pdb.set_trace()
    if isinstance(image_file, str):
        image_file = sitk.ReadImage(image_file)
    elif type(image_file) is not sitk.SimpleITK.Image:
        assert False, "Unknown data type to respaceing!"
    
    if isinstance(target_file, str):
        target_file = sitk.ReadImage(target_file)
    elif type(target_file) is not sitk.SimpleITK.Image:
        assert False, "Unknown data type to respaceing!"

    # set target size
    target_origing   = target_file.GetOrigin()
    target_direction = target_file.GetDirection()
    target_spacing   = target_file.GetSpacing()
    target_size      = target_file.GetSize()

    # pdb.set_trace()
    # initialize resampler
    resampler_image = sitk.ResampleImageFilter()
    # set the parameters of image
    resampler_image.SetReferenceImage(image_file)  # set rasampled image meta data same to origin data
    resampler_image.SetOutputOrigin(target_origing)
    resampler_image.SetOutputDirection(target_direction)  # set target image space
    resampler_image.SetOutputSpacing(target_spacing)  # set target image space
    resampler_image.SetSize(target_size)  # set target image size
    if resample_method == sitk.sitkNearestNeighbor:
        resampler_image.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler_image.SetOutputPixelType(sitk.sitkFloat32)
    resampler_image.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler_image.SetInterpolator(resample_method)

    # launch the resampler
    resampled_image_file = resampler_image.Execute(image_file)
    # pdb.set_trace()

    return resampled_image_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True, help="Path to the raw data directory")
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    dataset_ISLES22 = ISLES22(raw_data_dir)
    dataset_ISLES22.load_data()
    task_name = "Task500_Ischemic_Stroke_Test"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    
    maybe_mkdir_p(target_base)
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)

    dwi_file = respacing_file(dataset_ISLES22.dwi_path, target_spacing=[1, 1, 1], resample_method=sitk.sitkLinear)
    adc_file = reimplement_resize(dataset_ISLES22.adc_path, target_file=dwi_file, resample_method=sitk.sitkLinear)

    sitk.WriteImage(dwi_file,   join(target_imagesTs, 'ISLES22_' + '0001' + '_0000.nii.gz'))
    sitk.WriteImage(adc_file,   join(target_imagesTs, 'ISLES22_' + '0001' + '_0001.nii.gz'))

    generate_dataset_json(output_file=join(target_base, 'dataset.json'),
                          imagesTr_dir=target_imagesTr, 
                          imagesTs_dir=target_imagesTs, 
                          modalities=('dwi', 'adc'),
                          labels={0:'background', 1: 'Ischemic Stroke'}, 
                          dataset_name=task_name, 
                          sort_keys=True, 
                          license="ISLES22 license")


