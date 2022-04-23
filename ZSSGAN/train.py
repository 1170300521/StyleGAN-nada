'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024 
                                           --batch 2 
                                           --n_sample 4 
                                           --output_dir /path/to/output/dir 
                                           --lr 0.002 
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                           --iter 301 
                                           --source_class "photo" 
                                           --target_class "sketch" 
                                           --lambda_direction 1.0 
                                           --lambda_patch 0.0 
                                           --lambda_global 0.0 
                                           --lambda_texture 0.0 
                                           --lambda_manifold 0.0 
                                           --phase None 
                                           --auto_layer_k 0 
                                           --auto_layer_iters 0 
                                           --auto_layer_batch 8 
                                           --output_interval 50 
                                           --clip_models "ViT-B/32" "ViT-B/16" 
                                           --clip_model_weights 1.0 1.0 
                                           --mixing 0.0
                                           --save_interval 50
'''

import os
import numpy as np
from sklearn.utils import shuffle

import torch

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json
from sklearn.linear_model import SGDClassifier, SGDOneClassSVM

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True

def train(args):

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    args.svm_boundary = torch.randn(1, args.num_keep_first*512+1, device=device)
    args.svm_boundary = args.svm_boundary / args.svm_boundary.norm()
    args.svm_boundary.requires_grad=True

    # Set up output directories.
    prefix = f"{args.psp_loss_type}_{args.delta_w_type}"
    base_dir = f"partial_{args.lambda_partial}-content_{args.lambda_content}-clip-sample"
    if args.psp_model_weight > 0:
        sample_dir = os.path.join(args.output_dir, f"{prefix}_{args.num_keep_first}-alpha_{args.psp_alpha}-{base_dir}")
    else:
        sample_dir = os.path.join(args.output_dir, base_dir)

    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    # Training loop
    fixed_z = torch.randn(args.n_sample, 512, device=device)

    for i in tqdm(range(args.iter)):

        net.train()

        sample_z = mixing_noise(args.batch, 512, args.mixing, device)

        [sampled_src, sampled_dst], loss, psp_codes = net(sample_z, iters=i, return_psp_codes=True)

        net.zero_grad()
        
        loss.backward()

        if i > 10 and \
            args.psp_loss_type == "dynamic" and args.delta_w_type == 'svm':
            # TODO: LBFGS optimizer
            # def closure():
            #     svm_loss = net.psp_loss_model.svm_loss(psp_codes[0], psp_codes[1])
            #     w_optim.zero_grad()
            #     svm_loss.backward()
            #     print(f"SVM loss: {svm_loss}")
            #     return svm_loss
            #     # args.svm_boundary.grad.zero_()
            # w_optim.step(closure)
            # TODO: Adam Optimizer
            def closure(iters=30, tol_threh=0.1):
                prev_loss = 0
                for j in range(iters):
                    svm_loss = net.psp_loss_model.svm_loss(psp_codes[0], psp_codes[1])
                    w_optim.zero_grad()
                    svm_loss.backward()
                    print(f"SVM loss: {svm_loss}")
                    w_optim.step()
                    if torch.abs(svm_loss - prev_loss) < tol_threh:
                        break
                    prev_loss = svm_loss
            closure(iters=50, tol_threh=0.1)

        g_optim.step()
        
        # Embed the boundary of svm into training after sliding_window_size iterations
        if i == 10 and args.psp_loss_type == "dynamic" and args.delta_w_type == 'svm':
            w_optim = torch.optim.Adam(
                [args.svm_boundary],
                lr=0.005,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
            )

        tqdm.write(f"Clip loss: {loss}")

        if i % args.output_interval == 0:
            net.eval()
            if args.crop_for_cars:
                args.sample_truncation = 0.5
            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

                if args.crop_for_cars:
                    sampled_dst = sampled_dst[:, :, 64:448, :]

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)
                avg_image = net(net.generator_frozen.mean_latent.unsqueeze(0),
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=False,)[0]
                save_images(avg_image[1], sample_dir, "mean_w", 1, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

    for i in range(args.num_grid_outputs):
        net.eval()

        with torch.no_grad():
            sample_z = mixing_noise(16, 512, 0, device)
            [sampled_src, sampled_dst], _ = net(sample_z, truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")

    # Save conditional mask if exists
    if net.has_psp_loss:
        cond_mask, delta_w = net.psp_loss_model.get_conditional_mask()
        cond_mask = cond_mask.cpu().numpy()
        np.save(os.path.join(sample_dir, "cond_mask.npy"), cond_mask)

        if args.psp_loss_type == "dynamic":
            delta_w = delta_w / delta_w.norm()
            tmp = torch.zeros(18, 512, device=delta_w.device)
            tmp[0:args.num_keep_first] = delta_w.view(-1, 512)
            np.save(os.path.join(sample_dir, "dynamic_w.npy"), tmp.cpu().numpy())

        import matplotlib.pyplot as plt
        iter_num = len(net.psp_loss_model.iter_diff)
        x = np.arange(iter_num, dtype=np.float)
        # Save difference between the dynamic and overall
        plt.plot(x, net.psp_loss_model.iter_diff)
        plt.savefig(os.path.join(sample_dir, "mask_diff.png"))
        plt.clf()
        # Save dynamic mean values
        plt.plot(x, net.psp_loss_model.iter_mean)
        plt.savefig(os.path.join(sample_dir, "mask_mean.png"))
        plt.clf()
        # Save dynamic cosine similarity
        plt.plot(x, net.psp_loss_model.iter_sim)
        plt.savefig(os.path.join(sample_dir, 'mask_sim.png'))
        plt.clf()


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    dataset_size = {
        'ffhq': 1024,
        'cat': 512,
        'dog': 512,
        'church': 256,
        'horse': 256,
        'car': 512,
    }
    # save snapshot of code / args before training.
    args.output_dir = os.path.join("../paper_results", "demo_" + args.dataset, \
        args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_"), \
            args.output_dir)
    args.size = dataset_size[args.dataset]
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))

    img_type = args.style_img_dir.split(".")[-1]
    shutil.copy2(args.style_img_dir, os.path.join(args.output_dir, f'target.{img_type}'))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    