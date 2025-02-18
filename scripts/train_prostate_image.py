"""
Train a diffusion model on brats images.
"""

import argparse
import torch as th,os,sys
# print(os.getcwd())
# print("sys path",sys.path)
sys.path.append("/media/Data16T/Kishore/PALETTE/ddm_for_contrast/guided-diffusion_main/")


import guided_diffusion_in
# print("guided dif imported")
from guided_diffusion_in import dist_util, logger
# from guided_diffusion_in.load_prostate import SliceData
##
# from guided_diffusion_in.prostate_loader_5tp import SliceData
##
# from guided_diffusion_in.data_loader_3tp import SliceData

# for icassp review

from guided_diffusion_in.prostate_loader_icassp import SliceData
# from guided_diffusion_in.load_brats_all_modalities import load_data

from guided_diffusion_in.resample import create_named_schedule_sampler


from guided_diffusion_in.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_in.train_util_prostate import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("printing the model last",model.out)
    
    model.to(dist_util.dev())
    print("printing difusion",diffusion)
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    print("schedule sampler",args.schedule_sampler,schedule_sampler)
    
    
    ########### Creating Loader ############ 
    
    ##### prostate #######
    logger.log("creating data loader...for prostate")
    dataset = SliceData(root = args.data_dir,mode=args.mode)
    print("sample size check from loader",dataset.__getitem__(10)[0].shape)
    prostate_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size ,\
                                              shuffle = True ,num_workers= 1)
    print("len of loader",len(prostate_loader),"b X len",len(prostate_loader)*args.batch_size)
    data_loader = iter(prostate_loader)
    # a= next(data_loader)
    # print(a)

    ###### BRATS ########
    
    # dataset = load_data(brats_path=args.data_dir,h5_path=args.h5_path)
    # print("len dataset",len(dataset))
    # brats_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True ,num_workers= 1)
    # print("len loader",len(brats_loader))
    # data_loader= iter(brats_loader)
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data_loader=data_loader,#data_loader
        prostate_loader = prostate_loader,#brats_loader
        tf_logger_dir = args.tf_logger_dir,
        class_cond = args.class_cond,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        h5_path = "",
        mode = "",
        tf_logger_dir = "",
        schedule_sampler="uniform",
        lr=1e-4,
        class_cond = False, # exlicitly added class_cond=False
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
