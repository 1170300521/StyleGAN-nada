import os
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from utils.file_utils import save_images
from options.train_options import TrainOptions
from model.ZSSGAN import ZSSGAN
from model.sg2_model import Generator
from model.psp import pSp

SAVE_SRC = True
SAVE_DST = True
device = 'cuda'
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
# args.output_dir = os.path.join("../results", "demo_" + args.dataset, \
#     args.source_class.replace(" ", '_') + "+" + args.target_class.replace(" ", "_"), \
#         args.output_dir)
args.output_dir = '../results/tmp'
os.makedirs(args.output_dir, exist_ok=True)

# Make preprocess
resize = (192, 256) if args.dataset == 'car' else (256, 256)
preprocess = transforms.Compose([transforms.Resize(resize),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_transfer_image(img_path):
    psp_encoder = pSp(args.psp_path, device, args.size, has_decoder=True)
    psp_encoder.to(device)
    psp_encoder.requires_grad_(False)

    net = Generator(args.size, style_dim=512, n_mlp=8,)
    net.to(device)

    checkpoint = torch.load(args.frozen_gen_ckpt, map_location=device)
    net.load_state_dict(checkpoint['g_ema'], strict=True)

    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device).float()

    with torch.no_grad():
        net.eval()
        psp_encoder.eval()
        code, invert_img = psp_encoder(img)
        trans_img, _ = net([code],
                        input_is_latent=True,
                        randomize_noise=True,
                        return_latents=True)
    prefix = os.path.basename(img_path).split(".")[0]
    save_images(invert_img, args.output_dir, f'{prefix}_invert', 1, 1)
    save_images(trans_img, args.output_dir, f'{prefix}_transfer', 1, 1)

def get_inversion_img(args, img_path):
    psp_encoder = pSp(args.psp_path, device, args.size, has_decoder=True)
    psp_encoder.to(device)
    psp_encoder.requires_grad_(False)
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device).float()
    with torch.no_grad():
        code, invert_img = psp_encoder(img)
    save_images(invert_img, args.output_dir, 'invert', 1, 1)

if __name__ == "__main__":
    image_path = "../img/tmp/tmp.png"
    get_transfer_image(image_path)
