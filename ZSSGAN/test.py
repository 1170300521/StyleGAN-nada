import os
import numpy as np
import torch
import torchvision.transforms as transforms
import dlib
from pathlib import Path
from PIL import Image
from argparse import Namespace

from restyle.utils.common import tensor2im
from restyle.models.psp import pSp
from restyle.models.e4e import e4e
from restyle.utils.inference_utils import run_on_batch
from restyle.scripts.align_faces_parallel import align_face
from utils.file_utils import save_images
from options.train_options import TrainOptions
from model.ZSSGAN import ZSSGAN


args = TrainOptions().parse()
pretrained_model_dir = '../weights'
encoder_type = 'e4e' #@param['psp', 'e4e']
da_dir = os.path.join(args.output_dir, 'checkpoint')

restyle_experiment_args = {
    "model_path": os.path.join(pretrained_model_dir, f"restyle_{encoder_type}_ffhq_encode.pt"),
    "transform": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

model_path = restyle_experiment_args['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']

opts['checkpoint_path'] = model_path
opts = Namespace(**opts)

restyle_net = (pSp if encoder_type == 'psp' else e4e)(opts)

restyle_net.eval()
restyle_net.cuda()
print('Model successfully loaded!')

def run_alignment(image_path):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_transfer_image(image_path, net):

    input_image = run_alignment(image_path)

    img_transforms = restyle_experiment_args['transform']
    transformed_image = img_transforms(input_image)

    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolution


    with torch.no_grad():
        avg_image = get_avg_image(restyle_net)
        result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), restyle_net, opts, avg_image)

    #@title Convert inverted image.
    inverted_latent = torch.Tensor(result_latents[0][4]).cuda().unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        net.eval()
        
        [sampled_src, sampled_dst] = net(inverted_latent, input_is_latent=True)[0]
        
        # joined_img = torch.cat([sampled_src, sampled_dst], dim=0)
        save_images(sampled_dst, args.output_dir, Path(image_path).stem, 2, 0)
        # display(Image.open(os.path.join(sample_dir, f"joined_{str(0).zfill(6)}.jpg")).resize((512, 256)))

if __name__ == "__main__":
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)
    image_path = "../img/8.jpg"
    get_transfer_image(image_path, net)
