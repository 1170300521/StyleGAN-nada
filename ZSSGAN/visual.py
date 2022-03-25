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
import pickle
import torch

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from utils.file_utils import save_images
from utils.training_utils import mixing_noise
from utils.svm import get_delta_w

from model.ZSSGAN import ZSSGAN
from model.psp import pSp
from options.train_options import TrainOptions
from criteria.clip_loss import CLIPLoss


#TODO convert these to proper args
SAVE_SRC = True
SAVE_DST = True
device = 'cuda'


def get_clip_samples(args, n_samples=10000, debug=False):
    '''
    Sample images from GAN and embed them into clip space (norm)
    '''

    sample_dir = args.output_dir
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    net.eval()
    samples_vec = {k: [] for k in net.clip_loss_models.keys()}
    with torch.no_grad():
        for i in tqdm(range(n_samples)):
            sample_z = mixing_noise(1, 512, args.mixing, device)
            [sampled_src, _], _ = net(sample_z, truncation=args.sample_truncation)
            for k in net.clip_loss_models.keys():
                img_feats = net.clip_loss_models[k].get_image_features(sampled_src)
                img_feats = img_feats.squeeze().detach().cpu().numpy()
                samples_vec[k].append(img_feats)
            if debug:
                save_images(sampled_src, args.output_dir, 'sample', 1, i)
    
    for k in samples_vec.keys():
        with open(os.path.join(sample_dir, f'{args.dataset}_{k[-2::]}_samples.pkl'), 'wb') as f:
            pickle.dump(samples_vec[k], f)

def get_psp_codes(args, n_samples=10000, debug=False):
    '''
    Sample images from GAN and embed them into w+ space
    '''
    print("Initializing networks...")
    net = ZSSGAN(args)
    A_codes = []

    net.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples // 2)):
            sample_z = mixing_noise(2, 512, args.mixing, device)
            [sampled_src, _], _ = net(sample_z)
            img = net.psp_loss_model.psp_preprocess(sampled_src)
            codes, invert_img = net.psp_loss_model.get_image_features(img, norm=False)
            codes = codes.detach().cpu().numpy()
            A_codes.append(codes)

            if debug:
                save_images(sampled_src, args.output_dir, 'src', 2, i)
                save_images(invert_img, args.output_dir, 'invert', 2, i)

    np.save(f'../weights/psp_source/{args.dataset}_A_gen_w.npy', np.concatenate(A_codes, axis=0))

def show_w_edit(args):

    sample_dir = args.output_dir
    img_dir = os.path.join(sample_dir, f'{args.delta_w_type}_edit_imgs')
    os.makedirs(img_dir, exist_ok=True)
    
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    # Get editing vector (delta_w)
    if os.path.exists(os.path.join(sample_dir, f'{args.delta_w_type}_w.npy')):
        delta_w = np.load(os.path.join(sample_dir, f'{args.delta_w_type}_w.npy'))
        # delta_w = np.load(os.path.join(sample_dir, 'dynamic_svm_10-alpha_0.4-clip+psp-sample/dynamic_w.npy'))
        delta_w = torch.from_numpy(delta_w).unsqueeze(0).float().to(device)
    else:
        delta_w = None
        print(f"There not exists file namely {os.path.join(sample_dir, f'{args.delta_w_type}_w.npy')}")

    # Training loop
    net.eval()
    for i in range(10):
        with torch.no_grad():
            fixed_z = torch.randn(1, 512, device=device)
            [sampled_src, _], _ = net([fixed_z], truncation=args.sample_truncation, delta_w=delta_w)
            if args.crop_for_cars:
                sampled_src = sampled_src[:, :, 64:448, :]
            grid_rows = 1
            save_images(sampled_src, img_dir, "src", grid_rows, i)


def find_most_similar_imgs(args):
    src_sample_path = '../results/ffhq/photo+Original_samples/samples.pkl'
    clip_loss = CLIPLoss('cuda', 
                        lambda_direction=args.lambda_direction, 
                        lambda_patch=args.lambda_patch, 
                        lambda_global=args.lambda_global, 
                        lambda_manifold=args.lambda_manifold, 
                        lambda_texture=args.lambda_texture,
                        clip_model=args.clip_models[0],
                        args=args)
    with open(src_sample_path, 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
    if args.style_img_dir is not None:
        text = clip_loss.get_raw_img_features(args.style_img_dir).detach().cpu().numpy()[0]
    else:
        text = clip_loss.get_text_features(args.target_class).cpu().numpy()[0]
    sim = np.dot(X, text)
    orders = np.argsort(sim)[::-1]
    print(orders[0:20])
    print(sim[orders[0:20]])


def get_ffhq_codes(args):
    psp_encoder = pSp(args.psp_path, device, has_decoder=False)
    psp_encoder.to(device)
    psp_encoder.requires_grad_(False)
    ffhq_dir = "/home/ybyb/Dataset/ffhq_small"
    style_codes = []
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for img_path in tqdm(os.listdir(ffhq_dir)):
        img = Image.open(os.path.join(ffhq_dir, img_path)).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device).float()

        code, _ = psp_encoder(img)
        # code, invert_img = psp_encoder(img)
        # save_images(invert_img, args.output_dir, 'invert', 1, 1)
        style_codes.append(code.detach().cpu().numpy())
    style_codes = np.concatenate(style_codes, axis=0)
    
    with open(os.path.join(args.output_dir, "ffhq_w+.pkl"), 'wb') as f:
        pickle.dump(style_codes, f)


def get_pair_codes(args, n_samples=500):
    '''
    Generate pair w+ codes for images from domainA and domainB
    '''
    # Set up networks, optimizers.
    print("Initializing networks...")
    if os.path.exists(os.path.join(sample_dir, 'checkpoint', '000300.pt')):
        args.train_gen_ckpt = os.path.join(sample_dir, 'checkpoint', '000300.pt')
        print("Use pretrained weights from {}".format(os.path.join(sample_dir, 'checkpoint', '000300.pt')))
    net = ZSSGAN(args)

    A_codes = []
    B_codes = []
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples // 2)):
            sample_z = mixing_noise(2, 512, args.mixing, device)
            [sampled_src, sampled_dst], loss = net(sample_z)
            img = torch.cat([sampled_src, sampled_dst], dim=0)
            img = net.psp_loss_model.psp_preprocess(img)
            codes, invert_img = net.psp_loss_model.get_image_features(img, norm=False)
            codes = codes.detach().cpu().numpy()
            A_codes.append(codes[0:2])
            B_codes.append(codes[2:])
            # save_images(invert_img, args.output_dir, 'invert', 2, i)
            # save_images(sampled_src, args.output_dir, 'src', 2, i)
            # save_images(sampled_dst, args.output_dir, 'dst', 2, i)
    # np.save(os.path.join(args.output_dir, 'A_codes.npy'), np.concatenate(A_codes, axis=0))
    np.save(os.path.join(args.output_dir, 'B_codes.npy'), np.concatenate(B_codes, axis=0))

    get_delta_w(os.path.join(args.output_dir, 'B_codes.npy'), \
            neg_path=f"../weights/psp_source/{args.dataset}_A_gen_w.npy", \
            output_path=os.path.join(args.output_dir, f'{args.delta_w_type}_w.npy'), \
            delta_w_type=args.delta_w_type,
            args=args)
    

if __name__ == "__main__":
    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)
    
    args = TrainOptions().parse()
    
    dataset_size = {
        'ffhq': 1024,
        'cat': 512,
        'dog': 512,
        'church': 256,
        'horse': 256,
        'car': 512,
    }
    args.size = dataset_size[args.dataset]

    # Make output directory
    args.output_dir = os.path.join("../results", "demo_" + args.dataset, \
        args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_"), \
            args.output_dir)
    sample_dir = args.output_dir
    os.makedirs(sample_dir, exist_ok=True)

    # Multi-stage training
    get_pair_codes(args)
    show_w_edit(args)

    # Prepare sample vectors for each category or generate samples
    # get_clip_samples(args)
    # get_psp_codes(args)

    # target_list = ["Van Goph painting", "Miyazaki Hayao painting", "Fernando Botero painting",\
    #     "3D render in the style of Pixar", "Disney Princess", "White Walker",\
    #         "Sketch", "Anime", "Watercolor art with thick brushstrokes"]