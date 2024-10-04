# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 22.12.2023

import os
import shutil
import subprocess
import nibabel as nib
import tempfile
import warnings
import numpy as np
from src.utils import convert_to_nii, print_run, print_ensemble_message, print_completed, extract_brain, get_img_shape, save_nii, check_gpu_memory
from concurrent.futures import ThreadPoolExecutor

class IslesEnsemble:
    def __init__(self):
        pass

    def predict_ensemble(self, ensemble_path, input_dwi_path, input_adc_path, output_path, input_flair_path=None,
                         skull_strip=False, fast=False, save_team_outputs=False, parallelize=True):
        ''' Runs the Isles'22 Ensemble algorithm.

        Inputs:

        ensemble_path: path to Isles'22 ensemble git repo

        input_dwi_path: path to DWI image in nifti format

        input_adc_path: path to ADC image in nifti format

        output_path: path where stroke lesion mask output is stored

        input_flair_path: optional, path to FLAIR image in nifti format

        skull_strip: flag indicating if skull stripping is required

        fast: flag for running only the isles22 winning algorithm for faster inference

        save_team_outputs: flag for saving individual team outputs

        parallelize: flag for running algorithms in parallel (default is True)
        '''

        # Assigning the parameters to self to be accessible throughout the class
        self.ensemble_path = ensemble_path
        self.original_dwi_path = input_dwi_path
        self.input_dwi_path = input_dwi_path
        self.input_adc_path = input_adc_path
        self.output_path = output_path
        self.input_flair_path = input_flair_path
        self.skull_strip = skull_strip
        self.fast = fast
        self.save_team_outputs = save_team_outputs
        self.tmp_out_dir = tempfile.mkdtemp(prefix="tmp", dir="/tmp")
        self.keep_tmp_files = True
        gpu_avail = check_gpu_memory()
        if parallelize and gpu_avail:                   # Unless intentional false, paralellize inference.
            self.parallelize = True
        else:
            self.parallelize = False

        print(self.tmp_out_dir)
        print_ensemble_message()

        self.load_images()
        self.check_images()
        self.extract_brain()
        #self.copy_input_data()
        self.inference()
        self.copy_output_clean()
        self.ensemble()

        print_completed(self.original_dwi_path)

    def check_images(self):
        # Check image dimensions and affine compatibility
        assert get_img_shape(self.input_adc_path) == 3, f"Error: ADC is not 3D"
        if self.input_flair_path is not None:
            assert get_img_shape(self.input_flair_path) == 3, f"Error: FLAIR is not 3D"

        dwi_nii = nib.load(self.input_dwi_path)
        dwi_shape = dwi_nii.shape

        # Deal with 3D/4D DWI
        if len(dwi_shape) != 3:
            if len(dwi_shape) == 4:
                if dwi_shape[-1] == 1 or dwi_shape[-1] == 2:
                    dwi_data = dwi_nii.get_fdata()[..., -1]
                    new_dwi_header = dwi_nii.header.copy()
                    new_dwi_header['dim'][0] = 3
                    new_dwi_header['dim'][4] = 1
                    save_nii(dwi_data, dwi_nii.affine, new_dwi_header, self.input_dwi_path)
                    print('DWI is 4D and contains 2 volumes. Assuming b1000 provided as last channel...')
                    print()
                else:
                    raise ValueError(f"DWI is 4D and contains {dwi_shape[-1]} volumes. Please provide a 3D volume.")
            else:
                raise ValueError(f"DWI is {len(dwi_shape)}D. Please provide a 3D volume.")

        # Check affine compatibility between DWI and ADC
        adc_nii = nib.load(self.input_adc_path)
        if not np.array_equal(np.asarray(dwi_nii.affine), np.asarray(adc_nii.affine)):
            warnings.warn("DWI and ADC have different affine matrices! Changing affines to DWI one.", UserWarning)
            nib.save(nib.Nifti1Image(adc_nii.get_fdata(), dwi_nii.affine, adc_nii.header), self.input_adc_path)
            print()

    def copy_input_data(self):
        # Prepare input files for the ensemble algorithm
        if os.path.exists(os.path.join(self.ensemble_path, 'input')):
            shutil.rmtree(os.path.join(self.ensemble_path, 'input'))

        os.makedirs(os.path.join(self.ensemble_path, 'input', 'images', 'dwi-brain-mri'))
        os.mkdir(os.path.join(self.ensemble_path, 'input', 'images', 'adc-brain-mri'))
        os.mkdir(os.path.join(self.ensemble_path, 'input', 'images', 'flair-brain-mri'))

        shutil.copyfile(self.input_dwi_path,
                        os.path.join(self.ensemble_path, 'input', 'images', 'dwi-brain-mri', 'dwi.nii.gz'))
        shutil.copyfile(self.input_adc_path,
                        os.path.join(self.ensemble_path, 'input', 'images', 'adc-brain-mri', 'adc.nii.gz'))
        if self.input_flair_path is not None:
            shutil.copyfile(self.input_flair_path,
                            os.path.join(self.ensemble_path, 'input', 'images', 'flair-brain-mri', 'flair.nii.gz'))

    def copy_output_clean(self):
        # Copy results
        if self.save_team_outputs:
            shutil.copytree(os.path.join(self.tmp_out_dir, 'output'),
                            os.path.join(self.output_path, 'output_teams'))

        if not self.keep_tmp_files:
            shutil.rmtree(self.tmp_out_dir)

    def load_images(self):
        # Convert input files to NIfTI if needed
        self.input_dwi_path, nii_flag = convert_to_nii(self.input_dwi_path, self.tmp_out_dir, 'dwi')
        self.input_adc_path, _ = convert_to_nii(self.input_adc_path, self.tmp_out_dir, 'adc')
        if self.input_flair_path is not None:
            self.input_flair_path, _ = convert_to_nii(self.input_flair_path, self.tmp_out_dir, 'flair')

        self.skull_strip = True if not nii_flag else self.skull_strip

    def extract_brain(self):
        # Code based on HD-BET
        # Credits to Isensee et al (HBM 2019)
        # Git repo: https://github.com/MIC-DKFZ/HD-BET

        if self.skull_strip:
            if self.input_flair_path is not None:
                if not self.fast:
                    print("Skull stripping FLAIR ...")
                    extract_brain(self.input_flair_path, os.path.join(self.tmp_out_dir, 'flair', 'flair_ss'))
                    self.input_flair_path = self.input_flair_path.replace('flair.nii.gz', 'flair_ss.nii.gz')

            print("Skull stripping DWI and ADC ...")
            extract_brain(self.input_dwi_path, os.path.join(self.tmp_out_dir, 'dwi', 'dwi_ss'), save_mask=1)
            self.input_dwi_path = self.input_dwi_path.replace('dwi.nii.gz', 'dwi_ss.nii.gz')
            dwi_msk_path = self.input_dwi_path.replace('dwi_ss.nii.gz', 'dwi_ss_mask.nii.gz')
            dwi_msk_nii = nib.load(dwi_msk_path)

            adc_obj = nib.load(self.input_adc_path)
            adc_data = adc_obj.get_fdata() * dwi_msk_nii.get_fdata()
            self.input_adc_path = self.input_adc_path.replace('adc.nii.gz', 'adc_ss.nii.gz')
            save_nii(adc_data, dwi_msk_nii.affine, dwi_msk_nii.header, self.input_adc_path)

    @staticmethod
    def run_command(command, cwd):
        subprocess.run(command, shell=True, cwd=cwd)

    def inference(self):
        # Prepare commands
        commands = []

        # SEALS Command
        print_run('SEALS')
        path_seals = os.path.join(self.ensemble_path, 'src', 'SEALS/')
        command_seals = f'./nnunet_launcher.sh {self.tmp_out_dir}'
        commands.append((command_seals, path_seals))

        if self.input_flair_path is not None and not self.fast:
            # NVAUTO Command
            print_run('NVAUTO')
            path_nvauto = os.path.join(self.ensemble_path, 'src', 'NVAUTO/')
            command_nvauto = f'python process.py --input_path {self.tmp_out_dir}'
            commands.append((command_nvauto, path_nvauto))

            # FACTORIZER Command
            print_run('SWAN')
            path_factorizer = os.path.join(self.ensemble_path, 'src', 'FACTORIZER/')
            command_factorizer = f'python process.py --input_path {self.tmp_out_dir}'
            commands.append((command_factorizer, path_factorizer))

        # Execute commands based on parallelization flag
        if self.parallelize:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.run_command, cmd, cwd) for cmd, cwd in commands]
                for future in futures:
                    future.result()
        else:
            for cmd, cwd in commands:
                self.run_command(cmd, cwd)

    def ensemble(self):
        # Ensembling results
        path_voting = self.ensemble_path
        command_voting = f'python ./src/majority_voting.py -i {self.tmp_out_dir} -o {self.output_path} '
        subprocess.call(command_voting, shell=True, cwd=path_voting)