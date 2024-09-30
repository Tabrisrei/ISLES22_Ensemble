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

#import warnings
# Initialize colorama for cross-platform support
init(autoreset=True)
columns = os.get_terminal_size().columns


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


#    shutil.




if __name__ == '__main__':
    convert_to_nii('dwi', '/home/edelarosa/Documents/datasets/dwi_dcm')