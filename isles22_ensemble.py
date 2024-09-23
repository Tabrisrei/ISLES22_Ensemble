# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 22.12.2023


import os
import shutil
import subprocess
from colorama import Fore, Style, init

# Initialize colorama for cross-platform support
init(autoreset=True)
columns = os.get_terminal_size().columns

def print_ensemble_message():
    # Aesthetic header
    citation_title = "If you are using The Isles'22 Ensemble algorithm, please cite the following work:"
    citation_text = "de la Rosa, E. et al. (2024) A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge. arXiv:2403.19425."

    # Center the citation by adding padding
    centered_title = citation_title.center(columns)
    centered_citation = citation_text.center(columns)

    # Print centered citation with bold and color formatting
    print(
        Fore.WHITE + '####################################################################################################'.center(
            columns))
    print(
        Fore.WHITE + '####################################################################################################'.center(
            columns))

    print(Fore.BLUE + centered_title)
    print(Fore.YELLOW + Style.BRIGHT + centered_citation)

    print(
        Fore.WHITE + '####################################################################################################'.center(
            columns))
    print(
        Fore.WHITE + '####################################################################################################'.center(
            columns))


def print_run(algorithm):
    print('Running {} algorithm ...'.format(algorithm))

def predict_ensemble(isles_ensemble_path,
                     input_dwi_path,
                     input_adc_path,
                     output_path,
                     input_flair_path=None,
                     fast=False,
                     save_team_outputs=False):

    ''' Runs the Isles'22 Ensemble algorithm.
    
    Inputs:
    
    isles_ensemble_path: path to isles'22 ensemble git repo
    
    input_dwi_path: path to DWI image in nifti format
    
    input_adc_path: path to ADC image in nifti format
    
    input_flair_path [Optional]: path to FLAIR image in nifti format

    Fast: Only runs the winner algorithm
    
    output_path: path where stroke lesion mask output is stored
    
    save_team_outputs: True for storing the non-ensembled, individual algorithm results
    
    ''' 
    
    # Create folders to store inputs & intermediate results.
    if os.path.exists(os.path.join(isles_ensemble_path, 'input')):
        shutil.rmtree(os.path.join(isles_ensemble_path, 'input'))

    os.makedirs(os.path.join(isles_ensemble_path, 'input', 'images', 'dwi-brain-mri'))
    os.mkdir(os.path.join(isles_ensemble_path, 'input', 'images', 'adc-brain-mri'))
    os.mkdir(os.path.join(isles_ensemble_path, 'input', 'images', 'flair-brain-mri'))

    shutil.copyfile(input_dwi_path, os.path.join(isles_ensemble_path, 'input', 'images', 'dwi-brain-mri', 'dwi.nii.gz'))
    shutil.copyfile(input_adc_path, os.path.join(isles_ensemble_path, 'input', 'images', 'adc-brain-mri', 'adc.nii.gz'))
    if input_flair_path is not None:
        shutil.copyfile(input_flair_path, os.path.join(isles_ensemble_path, 'input', 'images', 'flair-brain-mri', 'flair.nii.gz'))

    # Ensemble prediction.
    predict(isles_ensemble_path, input_dwi_path, input_flair_path, fast_run=fast)

    # Copy results
    if save_team_outputs:
        shutil.copytree(os.path.join(isles_ensemble_path, 'output_teams'), os.path.join(output_path, 'output_teams'))
    shutil.copytree(os.path.join(isles_ensemble_path, 'output'), os.path.join(output_path, 'ensemble_output'))

    # Remove intermediate files
    shutil.rmtree(os.path.join(isles_ensemble_path, 'output_teams'))
    shutil.rmtree(os.path.join(isles_ensemble_path, 'output'))
    if os.path.exists(os.path.join(isles_ensemble_path, 'input')):
        shutil.rmtree(os.path.join(isles_ensemble_path, 'input'))

def predict(isles_ensemble_path, input_dwi_path, input_flair_path, fast_run=False):

    # Run SEALS Docker (https://github.com/Tabrisrei/ISLES22_SEALS)
    # Contact person: Shengbo Gao (GTabris@buaa.edu.cn)
    print_ensemble_message()


    print_run('SEALS')
    path_seals = isles_ensemble_path + '/SEALS/'
    command_seals = path_seals
    command_seals += f'nnunet_launcher.sh'
    subprocess.run(command_seals, shell=True, cwd=path_seals)

    if input_flair_path is not None and not fast_run:

        # Run NVAUTO Docker (https://github.com/mahfuzmohammad/isles22)
        # Contact person: Md Mahfuzur Rahman Siddiquee (mrahmans@asu.edu)

        print_run('NVAUTO')
        path_nvauto = isles_ensemble_path + '/NVAUTO/'
        command_nvauto = f'python process.py'
        subprocess.run(command_nvauto, shell=True, cwd=path_nvauto)

        # Run SWAN Docker (https://github.com/pashtari/factorizer-isles22)
        # Contact person: Pooya Ashtari (pooya.ashtari@esat.kuleuven.be)

        print_run('SWAN')
        path_factorizer = isles_ensemble_path + '/FACTORIZER/'
        command_factorizer = f'python process.py'
        subprocess.run(command_factorizer, shell=True, cwd=path_factorizer)

    # Ensembling results.

    path_voting = isles_ensemble_path
    command_voting = f'python majority_voting.py -i output_teams/ -o output/images/stroke-lesion-segmentation/'
    subprocess.call(command_voting, shell=True, cwd=path_voting)
    print(Fore.GREEN + Style.BRIGHT + f'Finished: {input_dwi_path}')




