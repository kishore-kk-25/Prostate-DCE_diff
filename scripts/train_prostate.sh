##training script for prostate that takes train_prostae_image.py and train_util_prosate.py
# class cond was False
# if want to resume add a  argument --resume_checkpoint path/to/uncon........../modelXXXXX.pt
# --data_dir is required whereas --h5_path is not required


#      ORIGINALLY BELOW
# python ./scripts/train_prostate_image.py --data_dir /media/Data16T/MRI/datasets/ProstateX_dce_h5/ --h5_path /media/Data16T/MRI/datasets/ProstateX_dce_h5/ --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 15 --learn_sigma True --class_cond False --tf_logger_dir /media/Data16T/Kishore/PALETTE/ddm_for_contrast/guided-diffusion/kishore_checkpoints/ --resume_checkpoint /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/unconditional_2024_02_14/model160000.pt

# python ./scripts/train_prostate_image.py --data_dir /media/Data16T/MRI/datasets/ProstateX_dce_h5/ --h5_path /media/Data16T/MRI/datasets/ProstateX_dce_h5/ --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 15 --learn_sigma True --class_cond False --tf_logger_dir GAYATHRI 

# for icaasp review  
python ./scripts/train_prostate_image.py --data_dir /media/Data16T/MRI/datasets/PROSTATE_DATASETS/ProstateX_dce_h5_120/ --h5_path /media/Data16T/MRI/datasets/PROSTATE_DATASETS/ProstateX_dce_h5_120/ --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 15 --learn_sigma True --class_cond False --tf_logger_dir GAYATHRI 

# /media/Data16T/MRI/datasets/PROSTATE_DATASETS/ProstateX_dce_h5_120
# python ./scripts/train_prostate_image.py --data_dir /media/Data16T/MRI/datasets/ProstateX_h5_wktrans --h5_path /media/Data16T/MRI/datasets/ProstateX_h5_wktrans/ --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 15 --learn_sigma True --class_cond False --tf_logger_dir GAYATHRI  

# 5tp
# python ./scripts/train_prostate_image.py --data_dir /media/Data16T/MRI/datasets/PROSTATE_DATASETS/ProstateX_nii_raw_h5_unscaled/ProstateX_160X160/ --h5_path /media/Data16T/MRI/datasets/PROSTATE_DATASETS/ProstateX_nii_raw_h5_unscaled/ProstateX_160X160/ --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 12 --learn_sigma True --class_cond False --tf_logger_dir GAYATHRI  

## difference map generation using palette

# ! pwd
# python ./scripts/train_prostate_image.py --data_dir /media/Data16T/Kishore/Chandra/Oscar_data_H5 --h5_path /media/Data16T/Kishore/Chandra/Oscar_data_H5 --mode train --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 12 --learn_sigma True --class_cond False --tf_logger_dir GAYAT

## 5tp
# changing from 15 to 8 

#--resume_checkpoint /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/palette_20:20__28_04_2024__prosx_ktrans_training/model160000.pt


#--resume_checkpoint /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/palette_12:14__08_03_2024/model260000.pt

# --tf_logger_dir variable is modified inside script so dont mind it 

# /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/unconditional_2024_03_05 
# /media/Data16T/Kishore/PALETTE/ddm_for_contrast/guided-diffusion/kishore_checkpoints_MAR_5_U66_i4_o2/
# ---> CHANGES MADE IMAGE_SIZE -160 --NUM CHANNELS 160 --NUM_RES_BLOCKS 4

# --resume_checkpoint /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/unconditional_2024_02_14/model160000.pt
