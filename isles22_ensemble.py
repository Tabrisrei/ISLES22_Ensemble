import os
import shutil
import subprocess

# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 22.12.2023

def print_run(algorithm):
    print('#########################################################')
    print('#########################################################')
    print('Running {} algorithm ...'.format(algorithm))
    print('#########################################################')
    print('#########################################################')


def predict_ensemble(isles_ensemble_path, input_dwi_path, input_adc_path, input_flair_path, output_path, save_team_outputs=False):
    ''' Runs the Isles'22 Ensemble algorithm.
    
    Inputs:
    
    isles_ensemble_path: path to isles'22 ensemble git repo
    
    input_dwi_path: path to DWI image in nifti format
    
    input_adc_path: path to ADC image in nifti format
    
    input_flair_path: path to FLAIR image in nifti format
    
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
    shutil.copyfile(input_flair_path, os.path.join(isles_ensemble_path, 'input', 'images', 'flair-brain-mri', 'flair.nii.gz'))

    # Ensemble prediction.
    predict(isles_ensemble_path, input_dwi_path)

    # Copy results
    if save_team_outputs:
        shutil.copytree(os.path.join(isles_ensemble_path, 'output_teams'), os.path.join(output_path, 'output_teams'))
    shutil.copytree(os.path.join(isles_ensemble_path, 'output'), os.path.join(output_path, 'ensemble_output'))

    # Remove intermediate files
    shutil.rmtree(os.path.join(isles_ensemble_path, 'output_teams'))
    shutil.rmtree(os.path.join(isles_ensemble_path, 'output'))
    if os.path.exists(os.path.join(isles_ensemble_path, 'input')):
        shutil.rmtree(os.path.join(isles_ensemble_path, 'input'))

def predict(isles_ensemble_path, input_dwi_path):

    # Run SEALS Docker (https://github.com/Tabrisrei/ISLES22_SEALS)
    # Contact person: Shengbo Gao (GTabris@buaa.edu.cn)
     
    print_run('SEALS')
    path_seals = isles_ensemble_path + 'SEALS/'
    command_seals = path_seals
    command_seals += f'nnunet_launcher.sh'
    subprocess.run(command_seals, shell=True, cwd=path_seals)

    # Run NVAUTO Docker (https://github.com/mahfuzmohammad/isles22)
    # Contact person: Md Mahfuzur Rahman Siddiquee (mrahmans@asu.edu)
    
    print_run('NVAUTO')
    path_nvauto = isles_ensemble_path + 'NVAUTO/'
    command_nvauto = f'python process.py'
    subprocess.run(command_nvauto, shell=True, cwd=path_nvauto)

    # Run SWAN Docker (https://github.com/pashtari/factorizer-isles22)
    # Contact person: Pooya Ashtari (pooya.ashtari@esat.kuleuven.be)

    print_run('SWAN')
    path_factorizer = isles_ensemble_path + 'FACTORIZER/'
    command_factorizer = f'python process.py'
    subprocess.run(command_factorizer, shell=True, cwd=path_factorizer)

    # Ensembling results.

    print_run('majority voting')
    path_voting = isles_ensemble_path
    command_voting = f'python majority_voting.py -i output_teams/ -o output/images/stroke-lesion-segmentation/'
    subprocess.call(command_voting, shell=True, cwd=path_voting)
    print('Finished: ', input_dwi_path)
    print("#####################################")
    print("#####################################")
    print("If you are using The Isles'22 Ensemble algorithm, please cite the following work:")
    print("de la Rosa et al. BLABLABLA. arXiv preprint 2023.")
    print("#####################################")
    print("#####################################")




