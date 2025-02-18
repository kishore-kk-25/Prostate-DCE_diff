"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os,sys
sys.path.append("/media/Data16T/Kishore/PALETTE/ddm_for_contrast/guided-diffusion_main/")

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F


# from guided_diffusion_in.load_prostate import SliceData # dataloader
from guided_diffusion_in.prostate_loader_icassp import SliceData
# from guided_diffusion_in.load_PMRI import SliceData # dataloader for prostateMRI


# from guided_diffusion_in.load_brats_all_modalities import load_data 


from guided_diffusion_in import dist_util, logger
print("Sample_prostate_modified_py")
print("logger get dir",logger.get_dir())
# script_utils 
from guided_diffusion_in.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("printing the model",model)
    # print("diffusion",diffusion)
    # diffusion_params = sum(p.numel() for p in diffusion.parameters())
    # print("diffusion params",diffusion_params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # print("params",model.parameters())
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    # print(model)
    # print(f'******************model is in device {next(model.parameters())}****************')
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    ###### creating loader ############
    
    logger.log("creating data loader...")
    
    # ######### Prostate ############
    dataset = SliceData(root = args.data_dir,mode=args.mode,train_test_split=0.0) # test
    print(" len of dataset",len(dataset))
    ######### Prostate MRI ############
    # dataset = SliceData(root = args.data_dir,mode=args.mode,train_test_split=0.0) # test
    # print(len(dataset))

    

    if args.mode=='test':
        prostate_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size ,\
                                                  shuffle = False ,num_workers= 1)
    else:
        prostate_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size ,\
                                                  shuffle = True ,num_workers= 1)
        
    print("len of loader",len(prostate_loader))
    
    ########## Brats ###################
    # dataset = load_data(brats_path=args.data_dir,h5_path=args.h5_path)
    # brats_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = False ,num_workers= 1)
    
    data_loader= iter(prostate_loader)
    
#     loader = iter(brats_loader)
#     image = next(loader)
#     image_t1 = image['image_t1']
#     image_t2 = image['image_t2']
    
#     print(f"Input Label is {label}")
#     logger.log("loading classifier...")
#     classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
#     classifier.load_state_dict(
#         dist_util.load_state_dict(args.classifier_path, map_location="cpu")
#     )
#     classifier.to(dist_util.dev())
#     if args.classifier_use_fp16:
#         classifier.convert_to_fp16()
#     classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
#             print(f"selected {selected}")
#             print(f" grad shape {len(th.autograd.grad(selected.sum(), x_in))}")
#             print(f"classifier scale is {args.classifier_scale}")
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        # print("x shape in model_fn",x.shape)
        
#         assert y is not None
        # print(f" t in model_fn is {t}")
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    orignal_img = []
    target_img =[]
    condition_map =[]
    noisy_map =[]
    reverse_map =[]
    latent = []

    
    for idx,data in enumerate(prostate_loader):#brats_loader):
        print("idx",idx)
        # all_images = []
        # all_labels = []
        # orignal_img = []
        # target_img =[]
        # condition_map =[]
        # noisy_map =[]
        # reverse_map =[]
        # latent = []
        
#     while len(all_images) * args.batch_size < args.num_samples:
        
#         target_classes = th.randint(
#             low=1, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())

#         input_classes = th.randint(
#         low=0, high=NUM_CLASSES-1, size=(args.batch_size,), device=dist_util.dev())

        model_kwargs = {}
#
        model_kwargs['y'] = 1 #target_classes
    
        batch = data[0].cuda()
        label = data[1]
        
        if (args.class_cond):
            diffusion_kwargs = model_kwargs
        else:
            diffusion_kwargs = None
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # print("use dim",args.use_ddim)
        # print("mode fn print",model_fn)
        sample = sample_fn(
            model_fn,
            # (args.batch_size, 3, args.image_size, args.image_size), # changing 1-->2
            # (args.batch_size, 3, args.image_size, args.image_size), bfore nov22-2024
            (args.batch_size, 2, args.image_size, args.image_size), # on nov 22-2024 for icassp exp
            
            # input_contrast = batch[:,0:3,:,:],
            # input_contrast = batch[:,0:4,:,:], bfore nov22-2024
            input_contrast = batch[:,0:8,:,:], #for  icassp
            
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            diffusion_kwargs = diffusion_kwargs,
            cond_fn=None,
            device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
            )
        # print(sample.shape, 'is sample ')
        # print(len(outlist),outlist[0].shape,outlist[1].shape,outlist[2].shape,outlist[3].shape,)

        
        # for i,out in enumerate(outlist):
        #     np.savez(f'/media/Data16T/Kishore/PALETTE/ddm_for_contrast/results_prostate/outlist_self_att/outlist_{i}.npz',out.cpu().numpy())

        
#         sample, noisy_latent, original_image, extra,noisy_mid,reverse_mid = sample_fn(
#             model_fn,
#             (args.batch_size, 1, args.image_size, args.image_size),original_image = image_t2,
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#             diffusion_kwargs = diffusion_kwargs,
#             cond_fn=cond_fn,
#             device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
#         )
#         sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())] # dist  -Returns the number of processes in the current process group
        # print("gathered samples len",len(gathered_samples))
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
#         gathered_samples1 = [th.zeros_like(noisy_mid) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_samples1, noisy_mid)  # gather not supported with NCCL
#         noisy_map.extend([sample.cpu().numpy() for sample in gathered_samples1])
        
        
#         gathered_samples2 = [th.zeros_like(reverse_mid) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_samples2, reverse_mid)  # gather not supported with NCCL
#         reverse_map.extend([sample.cpu().numpy() for sample in gathered_samples2])
        
#         gathered_samples3 = [th.zeros_like(noisy_latent) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_samples3, noisy_latent)  # gather not supported with NCCL
#         latent.extend([sample.cpu().numpy() for sample in gathered_samples3])
        
        
        
#         gathered_original_image = [th.zeros_like(data) for _ in range(dist.get_world_size())]
#         print(f"gathered_original_image shape {gathered_original_image[0].shape} original_image {data.shape}")
#         dist.all_gather(gathered_original_image, data.to(device = dist_util.dev()) ) # gather not supported with NCCL
        
#         gathered_map = [th.zeros_like(extra) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_map, extra)  # gather not supported with NCCL
#         condition_map.extend([sample.cpu().numpy() for sample in gathered_map])
        
        
        # original = list(batch[:,0:4,:,:][None])
        original = list(batch[:,0:8,:,:][None])
        
        orignal_img.extend([sample.cpu().numpy() for sample in original[:]])
        
        
        # target_image = list(batch[:,4:,:,:][None])
        target_image = list(batch[:,8:,:,:][None])
        
        target_img.extend([sample.cpu().numpy() for sample in target_image[:]])
#         print(f"target {target_img[0].shape}")
        
#         gathered_noisy_latent = [th.zeros_like(noisy_latent) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_noisy_latent, noisy_latent)  # gather not supported with NCCL
#         latent_img.extend([sample.cpu().numpy() for sample in gathered_noisy_latent])


#         gathered_labels = [th.zeros_like(target_classes) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_labels, target_classes)
#         all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    
        # logger.log(f"created {len(all_images) * args.batch_size} samples")
    
        # if(idx==20):
        #     break
        # break
        
        
    # print("len all images",len(all_images))
    
    arr = np.concatenate(all_images, axis=0)
    # print('shapes',len(all_images),arr.shape)
        # break
#     arr = arr[: args.num_samples]
    
    arr_org = np.concatenate(orignal_img,axis=0)
#     arr_org = arr_org[: args.num_samples]
    
    arr_target = np.concatenate(target_img,axis=0)
#     arr_latent = arr_latent[: args.num_samples]
    
#     arr_cmap = np.concatenate(condition_map, axis=0)
#     arr_cmap = arr_cmap[: args.num_samples]
    
#     label_arr = np.concatenate(all_labels, axis=0)
#     label_arr = label_arr[: args.num_samples]
    
#     noisy_arr = np.concatenate(noisy_map,axis =0)
#     arr_noisy = noisy_arr[: args.num_samples]
    
#     reverse_arr = np.concatenate(reverse_map,axis=0)
#     arr_reverse = reverse_arr[: args.num_samples]
    
    
#     latent_arr = np.concatenate(latent,axis=0)
#     latent_arr = latent_arr[: args.num_samples]
    
    # date = args.model_path.split('/')[-2].split('unconditional_')[-1]
    date = args.model_path.split('/')[-2].split('palette_')[-1]
    print("date at which it gets saved",date)
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path =  os.path.join(logger.get_dir(), f"samples_{shape_str}_{date}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        
        out_path_original = os.path.join(logger.get_dir(),f"original_image_{date}.npz")
        logger.log(f"saving to {out_path_original}")
        np.savez(out_path_original, arr_org)
        
        out_path_target = os.path.join(logger.get_dir(), f"target_image_{date}.npz")
        logger.log(f"saving to {out_path_target}")
        np.savez(out_path_target, arr_target)
            
    #         out_path_map = os.path.join(logger.get_dir(), f"condition_map.npz")
    #         logger.log(f"saving to {out_path_map}")
    #         np.savez(out_path_map, arr_cmap, label_arr)
            
    #         out_path_noisy_map = os.path.join(logger.get_dir(), f"noisy_map.npz")
    #         logger.log(f"saving to {out_path_noisy_map}")
    #         np.savez(out_path_noisy_map, arr_noisy, label_arr)
            
    #         out_path_reverse_map = os.path.join(logger.get_dir(), f"reverse_map.npz")
    #         logger.log(f"saving to {out_path_reverse_map}")
    #         np.savez(out_path_reverse_map, arr_reverse, label_arr)
            
    #         out_path_latent_map = os.path.join(logger.get_dir(), f"latent_map.npz")
    #         logger.log(f"saving to {out_path_latent_map}")
    #         np.savez(out_path_latent_map, latent_arr, label_arr)
            
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        data_dir="",
        h5_path = "",
        mode = "",
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path=""
#         classifier_path="",
#         classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
#     defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
