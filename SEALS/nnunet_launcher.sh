#!/bin/bash

# Suppressing output of environment variable exports
export nnUNet_raw_data_base="data/nnUNet_raw_data_base" >/dev/null 2>&1
export nnUNet_preprocessed="data/nnUNet_preprocessed" >/dev/null 2>&1
export RESULTS_FOLDER="../weights/SEALS/nnUNet_trained_models" >/dev/null 2>&1
export nnUNet_n_proc_DA=24 >/dev/null 2>&1

# Suppressing dataset conversion
python nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py >/dev/null 2>&1

# Suppressing inference commands
CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold0 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 0 \
               -z \
               --overwrite_existing \
               --disable_postprocessing \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold1 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 1 \
               -z \
               --overwrite_existing \
               --disable_postprocessing \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold2 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 2 \
               -z \
               --overwrite_existing \
               --disable_postprocessing \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold3 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 3 \
               -z \
               --overwrite_existing \
               --disable_postprocessing \
               >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold4 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 4 \
               -z \
               --overwrite_existing \
               --disable_postprocessing \
               >/dev/null 2>&1

# Suppressing the python scripts for postprocessing
python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold0 \
                        -m preliminary_phase \
                        -f fold0 \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold1 \
                        -m preliminary_phase \
                        -f fold1 \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold2 \
                        -m preliminary_phase \
                        -f fold2 \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold3 \
                        -m preliminary_phase \
                        -f fold3 \
                        >/dev/null 2>&1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold4 \
                        -m preliminary_phase \
                        -f fold4 \
                        >/dev/null 2>&1

# Suppressing ensemble softmax
model_0=test_result_recover/preliminary_phase/fold0
model_1=test_result_recover/preliminary_phase/fold1
model_2=test_result_recover/preliminary_phase/fold2
model_3=test_result_recover/preliminary_phase/fold3
model_4=test_result_recover/preliminary_phase/fold4

python -m ensemble_predictions \
       -f $model_0 \
          $model_1 \
          $model_2 \
          $model_3 \
          $model_4 \
       -o test_ensemble/ \
       --npz \
       >/dev/null 2>&1

# Suppressing thresholding final output
python threshold_redirect.py \
                            -i test_ensemble/ \
                            -o ../output_teams/seals/images/stroke-lesion-segmentation/ \
                            >/dev/null 2>&1
