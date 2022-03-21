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
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

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


def get_samples(args, n_samples=10000):
    '''
    Sample images from GAN and embed them into clip representation space (norm)
    '''
    sample_dir = args.output_dir
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
        img_feats = net.clip_loss_models['ViT-B/16'].get_image_features(sampled_src)
        save_images(sampled_src, args.output_dir, 'sample', 1, i)
        # img_feats = torch.cat([img_feats], dim=0).detach().cpu().numpy()
        img_feats = img_feats.squeeze().detach().cpu().numpy()
        samples_vec.append(img_feats)
    
    if args.return_w_only:
        w_codes = np.concatenate(w_codes, 0)
        with open(os.path.join(sample_dir, 'sample_w_codes.pkl'), 'wb') as f:
            pickle.dump(w_codes, f)
    else:
        with open(os.path.join(sample_dir, f'{args.dataset}_samples.pkl'), 'wb') as f:
            pickle.dump(samples_vec, f)
        

def get_avg_image(args):
    print("Initializing networks...")
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
    sample_dir = args.output_dir
    img_dir = os.path.join(sample_dir, 'edit_imgs')
    os.makedirs(img_dir, exist_ok=True)
    # Set up networks, optimizers.
    print("Initializing networks...")
    if os.path.exists(os.path.join(sample_dir, 'checkpoint', '000300.pt')):
        args.train_gen_ckpt = os.path.join(sample_dir, 'checkpoint', '000300.pt')
        print("Use pretrained weights from {}".format(os.path.join(sample_dir, 'checkpoint', '000300.pt')))
    net = ZSSGAN(args)

    # Get editing vector (delta_w)
    if os.path.exists(os.path.join(sample_dir, f'{args.delta_w_type}_w.npy')):
        delta_w = np.load(os.path.join(sample_dir, f'{args.delta_w_type}_w.npy'))
        # delta_w = np.load(os.path.join(sample_dir, 'dynamic_svm_10-alpha_0.4-clip+psp-sample/dynamic_w.npy'))
        delta_w = torch.from_numpy(delta_w).unsqueeze(0).float().to(device)
    else:
        delta_w = None
    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    # Training loop
    net.eval()
    for i in range(10):
        with torch.no_grad():
            fixed_z = torch.randn(1, 512, device=device)
            [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation, delta_w=delta_w)
            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

            grid_rows = 1
            save_images(sampled_src, img_dir, "src", grid_rows, i)
     

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


def embedding_pair_imgs(img_dir):
    src_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/ffhq/samples.pkl'
    with open(src_sample_path, 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
        std = np.std(X, axis=0)
        max_ids = np.argsort(std)[482:512]
        dist = np.max(X, axis=0) - np.min(X, axis=0)
    pca = PCA(n_components=512)
    pca.fit(X)

    clip_loss = CLIPLoss('cuda', 
                        lambda_direction=args.lambda_direction, 
                        lambda_patch=args.lambda_patch, 
                        lambda_global=args.lambda_global, 
                        lambda_manifold=args.lambda_manifold, 
                        lambda_texture=args.lambda_texture,
                        clip_model=args.clip_models[0],
                        args=args)
    special_dims = set(range(0, 512))
    for i in range(1, 4):
        orig_vec = clip_loss.get_raw_img_features(os.path.join(img_dir, f"{i}.png")).detach().cpu().numpy()
        mod_vec = clip_loss.get_raw_img_features(os.path.join(img_dir, f"{i}{i}.png")).detach().cpu().numpy()
        # orig_vec = pca.transform(orig_vec)
        # mod_vec = pca.transform(mod_vec)
        sub = np.abs(mod_vec - orig_vec)[0]
        sub[sub <= std * 2] = 0
        # sub[sub <= np.sqrt(pca.explained_variance_) * 2] = 0
        x = np.arange(len(sub))
        plt.ylim(-0.3, 0.3)
        plt.xlabel('dimension')
        plt.ylabel('value')
        # plt.scatter(max_ids, np.zeros(len(max_ids))+0.25, c='b', marker="o")
        # plt.scatter(x, orig_vec, c='r', s=1)
        # plt.scatter(x, mod_vec, c='g', marker='+', s=1)
        plt.scatter(x, sub, c='g', marker='+')
        plt.plot(x, x-x, c='r')
        
        plt.show()


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


def visualize_img_pca_weights(args):
    args.alpha = 1.5
    clip_loss = CLIPLoss('cuda', 
                        lambda_direction=args.lambda_direction, 
                        lambda_patch=args.lambda_patch, 
                        lambda_global=args.lambda_global, 
                        lambda_manifold=args.lambda_manifold, 
                        lambda_texture=args.lambda_texture,
                        clip_model=args.clip_models[0],
                        args=args)
    if args.style_img_dir is not None:
        text = clip_loss.get_raw_img_features(args.style_img_dir).detach().cpu().numpy()
    else:
        text = clip_loss.get_text_features(args.target_class).cpu().numpy()
    src_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/ffhq/samples.pkl'
    tgt_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/mean_w_clip.pkl'
    tgt_sample_path = src_sample_path
    # tgt_sample_path = '/home/ybyb/CODE/StyleGAN-nada/results/ffhq/photo+Van_Goph_painting/supress_src_0-alpha_0/samples.pkl'
    with open(src_sample_path, 'rb') as f:
        X = pickle.load(f)
        X = np.array(X)
        std = np.std(X, axis=0)
        max_ids = np.argsort(std)[482:512]
        # Define a pca and train it
    pca = PCA(n_components=512)
    pca.fit(X)

    # Define a GMM and fit it
    gm = GaussianMixture(n_components=10)
    orders = np.argsort(std)[::-1][0:50]
    gm_x = X[:, orders]
    gm.fit(gm_x)

    if src_sample_path == tgt_sample_path:
        Y = X
    else:
        with open(tgt_sample_path, 'rb') as f:
            Y = pickle.load(f)
            Y = np.array(Y)

    # Get the standar deviation of samples and set threshold for each dimension
    threshold = np.sqrt(pca.explained_variance_) * args.alpha
    x = np.arange(len(threshold))
#    l1 = plt.plot(x, threshold, 'r', label='pos_threshold')
#    l2 = plt.plot(x, -threshold, 'r', label='neg_threshold')
    plt.legend()
    plt.ylim(-0.3, 0.3)
    plt.xlabel('dimension')
    plt.ylabel('value')
    # Y_pca = pca.transform(Y)
    Y_pca = Y
    np.random.shuffle(Y_pca)
    for i in range(100):
        plt.scatter(x, Y_pca[i], s=1)
    plt.scatter(x, text[0], marker="+")
    plt.scatter(max_ids, np.zeros(len(max_ids))+0.25, marker="o")
    plt.savefig(os.path.join(args.output_dir, "clip_space.jpg"))
    plt.show()


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
    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

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
            os.path.join(args.output_dir, f'{args.delta_w_type}_w.npy'), \
                delta_w_type=args.delta_w_type,
                args=args)
                # neg_path="/home/ybyb/CODE/StyleGAN-nada/results/invert/ffhq_w+.npy")
    

if __name__ == "__main__":

    args = TrainOptions().parse()

    # Make output directory
    args.output_dir = os.path.join("../results", "demo_" + args.dataset, \
        args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_"), \
            args.output_dir)
    sample_dir = args.output_dir
    os.makedirs(sample_dir, exist_ok=True)

    get_pair_codes(args)
    visual(args)
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
    
    # get_avg_image(args)
    # visualize_img_pca_weights(args)
    # embedding_pair_imgs("/home/ybyb/yms/emo/open")
    # find_most_similar_imgs(args)
    # get_ffhq_codes(args)