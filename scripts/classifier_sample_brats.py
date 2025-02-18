"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F


from guided_diffusion.load_brats_sampler import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
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
#     print("hello i am here")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
#     print(f'******************model is in device {next(model.parameters()).device}****************')
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    ###### creating loader ############
    
    dataset = load_data(
        brats_path=args.brats_path,
        h5_path = args.h5_path
      
    )
    brats_loader = th.utils.data.DataLoader(dataset,batch_size = args.batch_size ,\
                                              shuffle = True ,num_workers= 1)### added
    loader = iter(brats_loader)
    image = next(loader)
    image_t1 = image['image_t1']
    image_t2 = image['image_t2']
#     print(f"Input Label is {label}")
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

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
#         assert y is not None
#         print(f" t in model_fn is {t}")
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    orignal_img = []
    latent_img =[]
    condition_map =[]
    noisy_map =[]
    reverse_map =[]
    latent = []

    
#     for _,data,label in enumerate(brats_loader):
    while len(all_images) * args.batch_size < args.num_samples:
        
        target_classes = th.randint(
            low=1, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())

        input_classes = th.randint(
        low=0, high=NUM_CLASSES-1, size=(args.batch_size,), device=dist_util.dev())

        model_kwargs = {}
#
        model_kwargs['y'] = target_classes

        if (args.class_cond):
            diffusion_kwargs = model_kwargs
        else:
            diffusion_kwargs = None

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, noisy_latent, original_image, extra,noisy_mid,reverse_mid = sample_fn(
            model_fn,
            (args.batch_size, 1, args.image_size, args.image_size),original_image = image_t2,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            diffusion_kwargs = diffusion_kwargs,
            cond_fn=cond_fn,
            device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
        )
#         sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        gathered_samples1 = [th.zeros_like(noisy_mid) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples1, noisy_mid)  # gather not supported with NCCL
        noisy_map.extend([sample.cpu().numpy() for sample in gathered_samples1])
        
        
        gathered_samples2 = [th.zeros_like(reverse_mid) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples2, reverse_mid)  # gather not supported with NCCL
        reverse_map.extend([sample.cpu().numpy() for sample in gathered_samples2])
        
        gathered_samples3 = [th.zeros_like(noisy_latent) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples3, noisy_latent)  # gather not supported with NCCL
        latent.extend([sample.cpu().numpy() for sample in gathered_samples3])
        
        
        
#         gathered_original_image = [th.zeros_like(data) for _ in range(dist.get_world_size())]
#         print(f"gathered_original_image shape {gathered_original_image[0].shape} original_image {data.shape}")
#         dist.all_gather(gathered_original_image, data.to(device = dist_util.dev()) ) # gather not supported with NCCL
        
        gathered_map = [th.zeros_like(extra) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_map, extra)  # gather not supported with NCCL
        condition_map.extend([sample.cpu().numpy() for sample in gathered_map])
        
        
        original = list(image_t2)
        orignal_img.extend([sample.cpu().numpy() for sample in original])
        
        
        target_image = list(image_t1)
        latent_img.extend([sample.cpu().numpy() for sample in target_image])
        
#         gathered_noisy_latent = [th.zeros_like(noisy_latent) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_noisy_latent, noisy_latent)  # gather not supported with NCCL
#         latent_img.extend([sample.cpu().numpy() for sample in gathered_noisy_latent])


        gathered_labels = [th.zeros_like(target_classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, target_classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    arr_org = np.concatenate(orignal_img)
    arr_org = arr_org[: args.num_samples]
    
    arr_latent = np.concatenate(latent_img)
    arr_latent = arr_latent[: args.num_samples]
    
    arr_cmap = np.concatenate(condition_map, axis=0)
    arr_cmap = arr_cmap[: args.num_samples]
    
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    
    noisy_arr = np.concatenate(noisy_map,axis =0)
    arr_noisy = noisy_arr[: args.num_samples]
    
    reverse_arr = np.concatenate(reverse_map,axis=0)
    arr_reverse = reverse_arr[: args.num_samples]
    
    
    latent_arr = np.concatenate(latent,axis=0)
    latent_arr = latent_arr[: args.num_samples]
    
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
        
        out_path_original = os.path.join(logger.get_dir(), f"original_image.npz")
        logger.log(f"saving to {out_path_original}")
        np.savez(out_path_original, arr_org, label_arr)
        
        out_path_latent = os.path.join(logger.get_dir(), f"target_image.npz")
        logger.log(f"saving to {out_path_latent}")
        np.savez(out_path_latent, arr_latent, label_arr)
        
        out_path_map = os.path.join(logger.get_dir(), f"condition_map.npz")
        logger.log(f"saving to {out_path_map}")
        np.savez(out_path_map, arr_cmap, label_arr)
        
        out_path_noisy_map = os.path.join(logger.get_dir(), f"noisy_map.npz")
        logger.log(f"saving to {out_path_noisy_map}")
        np.savez(out_path_noisy_map, arr_noisy, label_arr)
        
        out_path_reverse_map = os.path.join(logger.get_dir(), f"reverse_map.npz")
        logger.log(f"saving to {out_path_reverse_map}")
        np.savez(out_path_reverse_map, arr_reverse, label_arr)
        
        out_path_latent_map = os.path.join(logger.get_dir(), f"latent_map.npz")
        logger.log(f"saving to {out_path_latent_map}")
        np.savez(out_path_latent_map, latent_arr, label_arr)
        

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        brats_path ="",
        h5_path = "",
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
