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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions
from criteria.clip_loss import CLIPLoss

#TODO convert these to proper args
SAVE_SRC = True
SAVE_DST = True
device = 'cuda'


def get_samples(args, n_samples=10000):
    '''
    Sample images from GAN and embed them into clip representation space (norm)
    '''
    # Set up output directories.
    desc = args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_")
    # prefix = f"supress_src_{args.supress_src}-alpha_{args.alpha}"

    args.output_dir = os.path.join("../results", args.dataset, desc)
    sample_dir = args.output_dir
    os.makedirs(sample_dir, exist_ok=True)
    # Set up networks, optimizers.
    print("Initializing networks...")
    if os.path.exists(os.path.join(sample_dir, 'checkpoint', '000300.pt')):
        args.frozen_gen_ckpt = os.path.join(sample_dir, 'checkpoint', '000300.pt')
        print("Use pretrained weights from {}".format(os.path.join(sample_dir, 'checkpoint', '000300.pt')))
    net = ZSSGAN(args)
    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    net.eval()
    samples_vec = []
    w_codes = []
    for i in tqdm(range(n_samples)):
        sample_z = mixing_noise(1, 512, args.mixing, device)
        if args.return_w_only:
            w_code = net(sample_z)
            w_codes.append(w_code[0].detach().cpu().numpy())
            continue
        [sampled_src, sampled_dst], loss = net(sample_z)
        img_feats = net.clip_loss_models['ViT-B/32'].get_image_features(sampled_src)
        # img_feats = torch.cat([img_feats], dim=0).detach().cpu().numpy()
        img_feats = img_feats.squeeze().detach().cpu().numpy()
        samples_vec.append(img_feats)
    
    if args.return_w_only:
        w_codes = np.concatenate(w_codes, 0)
        with open(os.path.join(sample_dir, 'sample_w_codes.pkl'), 'wb') as f:
            pickle.dump(w_codes, f)
    else:
        with open(os.path.join(sample_dir, 'samples.pkl'), 'wb') as f:
            pickle.dump(samples_vec, f)
        

def get_avg_image(args):
    print("Initializing networks...")
    args.output_dir = os.path.join("../results", "demo_" + args.dataset, \
        args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_"), \
            args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint', '000300.pt')):
        args.train_gen_ckpt = os.path.join(args.output_dir, 'checkpoint', '000300.pt')
        print("Use pretrained weights from {}".format(os.path.join(args.output_dir), 'checkpoint', '000300.pt'))
    net = ZSSGAN(args)
    avg_image = net(net.generator_frozen.mean_latent.unsqueeze(0),
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=False,)[0]
    save_images(avg_image[0], args.output_dir, 'src', 1, 0)
    save_images(avg_image[1], args.output_dir, 'dst', 1, 0)
    src_img_feats = net.clip_loss_models['ViT-B/32'].get_image_features(avg_image[0])
    # img_feats = torch.cat([img_feats], dim=0).detach().cpu().numpy()
    src_img_feats = src_img_feats.squeeze().detach().cpu().numpy()
    tgt_img_feats = net.clip_loss_models['ViT-B/32'].get_image_features(avg_image[1])
    # img_feats = torch.cat([img_feats], dim=0).detach().cpu().numpy()
    tgt_img_feats = tgt_img_feats.squeeze().detach().cpu().numpy()
    with open(os.path.join(args.output_dir, 'mean_w_clip.pkl'), 'wb') as f:
        pickle.dump([src_img_feats, tgt_img_feats], f)

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
     

def visualize_pca_weights(args):
    args.alpha = 1
    clip_loss = CLIPLoss('cuda', 
                        lambda_direction=args.lambda_direction, 
                        lambda_patch=args.lambda_patch, 
                        lambda_global=args.lambda_global, 
                        lambda_manifold=args.lambda_manifold, 
                        lambda_texture=args.lambda_texture,
                        clip_model=args.clip_models[0],
                        args=args)
    tgt_vec_list = clip_loss.get_text_features(args.target_class).cpu().numpy()  # Multiple vectors due to prompt engineer
    tgt_pca_list = clip_loss.pca.transform(tgt_vec_list)
    threshold = clip_loss.threshold
    x = np.arange(len(threshold))
    l1 = plt.scatter(x, tgt_pca_list[0], s=5)
    # l2 = plt.plot(x, tgt_pca_list.mean(0), 'b', label='target_mean')
    l3 = plt.plot(x, threshold, 'r', label='pos_threshold')
    l4 = plt.plot(x, -threshold, 'r', label='neg_threshold')
    speical_num = (np.abs(tgt_pca_list[0]) > threshold).sum()
    # plt.plot(x, tgt_pca_list[0], 'go-', x, tgt_pca_list.mean(0), 'b+-', x, threshold, 'r^-')
    plt.xlabel('dimension')
    plt.ylabel('value')
    plt.title(args.target_class + ", Special " + str(speical_num))
    plt.legend()
    plt.ylim(-0.3, 0.3)
    plt.savefig(os.path.join(args.output_dir, args.target_class.replace(" ", '_') + ".jpg"))
    plt.show()


def visualize_img_pca_weights(args):
    args.alpha = 1.5
    src_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/ffhq/samples.pkl'
    tgt_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/mean_w_clip.pkl'
    # tgt_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/ffhq/photo+Van_Goph_painting/supress_src_0-alpha_0/samples.pkl'
    with open(src_sample_path, 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
        # Define a pca and train it
    pca = PCA(n_components=512)
    pca.fit(X)

    if src_sample_path == tgt_sample_path:
        Y = X
    else:
        with open(tgt_sample_path, 'rb') as f:
            Y = pickle.load(f)
            Y = np.array(Y)

    # Get the standar deviation of samples and set threshold for each dimension
    threshold = np.sqrt(pca.explained_variance_) * args.alpha
    x = np.arange(len(threshold))
    l1 = plt.plot(x, threshold, 'r', label='pos_threshold')
    l2 = plt.plot(x, -threshold, 'r', label='neg_threshold')
    plt.legend()
    plt.ylim(-0.3, 0.3)
    plt.xlabel('dimension')
    plt.ylabel('value')
    Y_pca = pca.transform(Y)
    np.random.shuffle(Y_pca)
    for i in range(2):
        plt.scatter(x, Y_pca[i], s=1)
    plt.savefig(os.path.join(args.output_dir, "mean_w.jpg"))
    plt.show()


if __name__ == "__main__":

    args = TrainOptions().parse()
    # visual(args)
    # get_samples(args)
    # target_list = ["Van Goph painting", "Miyazaki Hayao painting", "Fernando Botero painting",\
    #     "3D render in the style of Pixar", "Disney Princess", "White Walker",\
    #         "Sketch", "Anime", "Watercolor art with thick brushstrokes"]
    # target_list = ["Van Goph painting", "Miyazaki Hayao painting", "Sketch"]
    # os.makedirs(args.output_dir, exist_ok=True)
    # for target in target_list:
    #     args.target_class = target
    #     visualize_pca_weights(args)
        # get_samples(args)
    
    get_avg_image(args)
    visualize_img_pca_weights(args)