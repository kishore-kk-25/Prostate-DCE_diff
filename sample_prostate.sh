python scripts/sample_prostate.py \
--model_path /srv/Data/playground_data/checkpoint_folder/brats_checkpoint/pallette/unconditional_2023_08_02/ema_0.9999_500000.pt \
--data_dir /srv/Data/playground_data/brats/brats_validation_data/MICCAI_BraTS2020_ValidationData \
--h5_path /srv/Data/playground_data/brats/h5_test_all_modalities \
--mode test \
--class_cond False --image_size 128 --learn_sigma True --num_channels 128 --num_res_blocks 3 --batch_size 1 --num_samples 1 --timestep_respacing 1000 --use_ddim False
    
    
### /srv/Data/playground_data/checkpoint_folder/prostate_checkpoint/pallette_prostate/unconditional-2023-07-10/ema_0.9999_400000.pt

#### /srv/Data/playground_data/prostate/

