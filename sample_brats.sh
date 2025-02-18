python scripts/classifier_sample_brats.py \
    --model_path /srv/Data/playground_data/checkpoint_folder/model_T1T2_conditional/Brats-2023-01-26/ema_0.9999_700000.pt \
    --classifier_path /srv/Data/playground_data/checkpoint_folder/T1T2_classifier_checkpoint/Brats-2023-01-16/model200000.pt \
    --brats_path /srv/Data/playground_data/brats/brats_training_data/data_raw \
    --h5_path /srv/Data/playground_data/brats/h5_t1_t2\
    --class_cond True --image_size 256 --learn_sigma True --num_channels 128 --num_res_blocks 3 --image_size 256 --classifier_attention_resolutions 8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 0 --classifier_use_fp16 False --batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True