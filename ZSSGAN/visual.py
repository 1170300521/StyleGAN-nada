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

import argparse
from math import pi
import os
import numpy as np

import torch
from torch._C import device

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json
import pickle

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

#TODO convert these to proper args
SAVE_SRC = True
SAVE_DST = True
device = 'cuda'


def get_samples(args, n_samples=10000):
    '''
    Sample images from GAN and embed them into clip representation space (norm)
    '''
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, 'orig_samples')
    os.makedirs(sample_dir, exist_ok=True)
    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    net.eval()
    samples_vec = []
    for i in tqdm(range(n_samples)):
        sample_z = mixing_noise(1, 512, args.mixing, device)
        [sampled_src, sampled_dst], loss = net(sample_z)
        img_feats = net.clip_loss_models['ViT-B/32'].get_image_features(sampled_src)
        # img_feats = torch.cat([img_feats], dim=0).detach().cpu().numpy()
        img_feats = img_feats.squeeze().detach().cpu().numpy()
        samples_vec.append(img_feats)

    with open(os.path.join(sample_dir, 'samples.pkl'), 'wb') as f:
        pickle.dump(samples_vec, f)
        
    
def visual(args):

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "vis_sample")

    os.makedirs(sample_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    # Training loop
    fixed_z = torch.randn(args.n_sample, 512, device=device)

    net.eval()

    with torch.no_grad():
        [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

        if args.crop_for_cars:
            sampled_dst = sampled_dst[:, :, 64:448, :]

        grid_rows = int(args.n_sample ** 0.5)

        if SAVE_SRC:
            save_images(sampled_src, sample_dir, "src", grid_rows, 0)

        if SAVE_DST:
            save_images(sampled_dst, sample_dir, "dst", grid_rows, 0)
     

if __name__ == "__main__":

    args = TrainOptions().parse()

    # visual(args)
    get_samples(args)
    
