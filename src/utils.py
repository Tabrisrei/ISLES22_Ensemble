# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 24.09.2024

import os
import glob
import shutil
from pathlib import Path
import subprocess
from colorama import Fore, Style, init
import textwrap
import nibabel as nib
import SimpleITK as sitk
import os
import requests
from matplotlib import pyplot as plt
import numpy as np
#import warnings
# Initialize colorama for cross-platform support
init(autoreset=True)
try:
    columns = os.get_terminal_size().columns
except:
    columns = 80

def print_completed(mypath):
    print(Fore.GREEN + Style.BRIGHT + f'Finished: {mypath}')


def print_ensemble_message():
    # Aesthetic header
    citation_title = "If you are using The Isles'22 Ensemble algorithm, please cite the following work:"
    citation_text = (
        "de la Rosa, E. et al. (2024) A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: "
        "Generalizability and Clinical Utility Beyond the ISLES Challenge. arXiv:2403.19425."
    )

    # Define the maximum width for each line in the terminal
    max_width = 120

    # Wrap the citation text
    wrapped_citation = textwrap.fill(citation_text, max_width)

    # Print the header with formatting
    print(Fore.WHITE + '#' * (max_width + 4))
    print(Fore.WHITE + '#' * (max_width + 4))
    print(Fore.BLUE + citation_title)

    # Print the citation with line breaks
    print(Fore.YELLOW + Style.BRIGHT + wrapped_citation)

    # Print the footer with formatting
    print(Fore.WHITE + '#' * (max_width + 4))
    print(Fore.WHITE + '#' * (max_width + 4))

def print_run(algorithm):
    print('Running {} algorithm ...'.format(algorithm))

def get_img_shape(image_path):
    myimg = nib.load(image_path)
    return len(myimg.shape)

def save_nii(mydata, myaffine, myheader, outpath):
    nib.save(nib.Nifti1Image(mydata, myaffine, myheader), outpath)

def convert_to_nii(input_path, tmp_dir, image_mod):
    # case dcm
    if Path(input_path).is_dir(): # dcm folder
        if any(Path(input_path).rglob('*.dcm')):
            output_dir = os.path.join(tmp_dir, image_mod)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            command = [
                'dcm2niix',  # The dcm2niix command
                '-z', 'y',  # Compress the output NIfTI files (.nii.gz)
                '-o', output_dir,  # Output directory
                input_path  # Directory containing DICOM files
            ]

            print('Converting {} dicom to nifti...'.format(image_mod))
            with open(os.devnull, 'w') as devnull:
                    subprocess.run(command, stdout=devnull, stderr=devnull, check=True)

            new_path = os.path.join(output_dir, '{}.nii.gz'.format(image_mod))
            os.rename(glob.glob(os.path.join(output_dir, '*.nii.gz'))[0], new_path)

            return new_path, False # flag to indicate dcm/nii

    # case .nii
    else:
        if input_path[-4:] == '.nii' or input_path[-7:] == '.nii.gz' or input_path[-4:] == '.mha':
            output_dir = os.path.join(tmp_dir, image_mod)
            output_dir_file = os.path.join(output_dir, image_mod+'.nii.gz')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.copyfile(input_path, output_dir_file)
            return output_dir_file, True # flag to indicate dcm/nii
        else:
            raise ValueError("No .nii, .nii.gz, .mha, or Dicom files available.")


def extract_brain(input_path, output_path, gpu=True, save_mask=0):
    if gpu:
        command_hd_bet = f'hd-bet -i {input_path} -o {output_path} -s {save_mask} -mode fast'
    else:
        command_hd_bet = f'hd-bet -i {input_path} -o {output_path} -s {save_mask} -device cpu -mode fast -tta 0'

    # Run HD-BET while suppressing warnings (stderr) but keeping print output (stdout)
    with open(os.devnull, 'w') as devnull:
        subprocess.call(command_hd_bet, shell=True, stderr=devnull)


def check_gpu_memory(min_free_memory_gb=12):
    try:
        # Run the `nvidia-smi` command to get GPU memory details
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse the output
        free_memory_list = result.stdout.strip().split("\n")
        free_memory_list = [int(mem) for mem in free_memory_list]  # Convert memory values to integers (in MB)

        # Check if any GPU has sufficient free memory
        for free_memory_mb in free_memory_list:
            free_memory_gb = free_memory_mb / 1024  # Convert MB to GB
            #print(free_memory_gb)
            if free_memory_gb >= min_free_memory_gb:
                return True

        return False

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while trying to check GPU memory: {e.stderr}")
        return False
#    shutil.


def register_mri(fixed_image_path, moving_image_path, out_dir_path, transformation='rigid'):
    # Set up the ElastixImageFilter
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)
    elastix.SetOutputDirectory(os.path.dirname(out_dir_path))

    elastix.SetParameterMap(sitk.GetDefaultParameterMap(transformation))
    elastix.LogToConsoleOff()
    elastix.LogToFileOff()
    #elastix.AddParameterMap(sitk.GetDefaultParameterMap("affine"))

    elastix.Execute()

    reg_image = elastix.GetResultImage()

    # Optionally, save the registered image
    sitk.WriteImage(reg_image, out_dir_path)
    # Execute the registration

def propagate_image(mask_image_path, out_dir_path, is_mask = False):
    mask_image = sitk.ReadImage(mask_image_path)
    transform_param_files = glob.glob(os.path.join(os.path.dirname(out_dir_path), 'TransformParameters.*.txt'))

    for param_file in transform_param_files:
        # Read the parameter map and set nearest neighbor interpolator for binary masks
        transform_param_map = sitk.ReadParameterFile(param_file)
        if is_mask:
            transform_param_map['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']

        # Apply the transformation using Transformix
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(mask_image)
        transformix.SetTransformParameterMap(transform_param_map)
        transformix.LogToConsoleOff()
        transformix.LogToFileOff()
        # Execute the transformation
        transformix.Execute()

        # Get the transformed mask image for the next iteration
        mask_image = transformix.GetResultImage()
    # Save the final transformed binary mask
    sitk.WriteImage(mask_image, out_dir_path)





def get_flair_atlas(output_path):
    '''    Get a FLAIR-MNI vascular territory atlas from https://zenodo.org/records/3379848
    https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.00208/full   '''

    url_flair = url = "https://zenodo.org/record/3379848/files/caa_flair_in_mni_template_smooth_brain_intres.nii.gz?download=1"

    # Download the file if it does not exist
    if not os.path.exists(output_path):
        print('Getting vascular territory atlas. If you use it, please cite:')
        print(' Schirmer, Markus D., et al. "Spatial signature of white matter hyperintensities in stroke patients." Frontiers in neurology 10 (2019): 208.')

        response = requests.get(url_flair)
        with open(output_path, 'wb') as f:
            f.write(response.content)


def registration_qc(image_paths, output_path, mask_path=None):
    # Load mask if provided
    if mask_path is not None:
        brain = nib.load(mask_path).get_fdata()
    else:
        brain = None

    # Load images
    images = [nib.load(img_path).get_fdata() for img_path in image_paths]

    # Set brain mask if mask not provided
    if brain is None:
        brain = 1.0 * (images[1] > 0.1)  # Use first image as mask reference
    brain[brain == 0] = np.nan  # Replace zeroes with NaN for transparency

    # Get the number of images
    num_images = len(images)

    # Calculate the central slice
    central_slice = round(images[0].shape[-1] / 2)

    # Set up the figure size dynamically based on number of images
    plt.figure(figsize=(5 * num_images, 5), dpi=80, facecolor='black')
    #plt.style.use("dark_background")
    plt.subplots_adjust(left=0.01,
                        bottom=0.01,
                        right=0.99,
                        top=0.99,
                        wspace=0.1,
                        hspace=0)

    # Plot each image with the mask overlay
    for i, img in enumerate(images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(np.rot90(img[:, :, central_slice]), 'gray')
        plt.imshow(np.rot90(brain[:, :, central_slice]), 'Accent', interpolation='none', alpha=0.3)
        plt.axis('off')

    # Show and save the figure
    plt.savefig(output_path)
    #plt.show()

    #plt.close('all')

if __name__ == '__main__':
    convert_to_nii('dwi', '/home/edelarosa/Documents/datasets/dwi_dcm')