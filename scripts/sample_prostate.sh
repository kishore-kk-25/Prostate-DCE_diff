# this is for sampling 
# created on feb 23-02-24

# ORIGINALLY WORKING 

# python scripts/sample_prostate_modified_loop.py --model_path /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/unconditional_2024_02_14/ema_0.9999_170000.pt  --data_dir  /media/Data16T/MRI/datasets/ProstateX_dce_h5  --mode test --class_cond False --image_size 128 --learn_sigma True --num_channels 128 --num_res_blocks 3 --batch_size 1 --num_samples 2 --timestep_respacing 1000 --use_ddim False

## FOR ICASSP_review
python scripts/sample_prostate_modified_loop.py --model_path /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/palette_13:14__26_11_2024__8chnl_120_data_re_training_for_icassp/model110000.pt --data_dir /media/Data16T/MRI/datasets/PROSTATE_DATASETS/divya_sample_test_files/  --mode test --class_cond False --image_size 128 --learn_sigma True --num_channels 128 --num_res_blocks 3 --batch_size 1 --num_samples 2 --timestep_respacing 1000 --use_ddim False


# FOR normal dce2,dce3 prediction ,the cvpr one

# python scripts/sample_prostate_modified_loop.py --model_path /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/palette_14:58__28_03_2024__ProsX_training_without_ADC_5chnl_imgs/model090000.pt  --data_dir  /media/Data16T/MRI/datasets/ProstateX_dce_h5  --mode test --class_cond False --image_size 128 --learn_sigma True --num_channels 128 --num_res_blocks 3 --batch_size 1 --num_samples 2 --timestep_respacing 1000 --use_ddim False


# python scripts/sample_prostate_modified_loop.py --model_path /media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/palette_15:27__01_05_2024__prosx_ktrans_retraining/model530000.pt  --data_dir  /media/Data16T/MRI/datasets/ProstateX_h5_wktrans  --mode test --class_cond False --image_size 128 --learn_sigma True --num_channels 128 --num_res_blocks 3 --batch_size 1 --num_samples 2 --timestep_respacing 1000 --use_ddim False

# --num_samples is not  used  anywhere