from turtle import forward
import torch
import torchvision.transforms as transforms

import numpy as np

from ZSSGAN.criteria.clip_loss import DirectionLoss
from ZSSGAN.model.psp import pSp

class PSPLoss(torch.nn.Module):
    def __init__(self, device, args=None):
        super(PSPLoss, self).__init__()

        self.device = device
        self.args = args
        self.model = pSp(self.args.psp_path, device, output_size=args.size, has_decoder=False)
        self.model.to(device)

        self.psp_preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].]
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.target_direction = self.get_target_direction()
        self.direction_loss = DirectionLoss('cosine')

    def get_target_direction(self, normalize=True):
        delta_w = np.load('/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/small_delta_w.npy')
        # delta_w = -np.load('/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/delta.npy').mean(0)
        # delta_w = np.load('/home/ybyb/CODE/StyleGAN-nada/results/invert/tmp.npy')
        # delta_w = delta_w[-1:]
        delta_w = torch.from_numpy(delta_w).to(self.device).float().view(1, -1)
        if normalize:
            delta_w /= delta_w.clone().norm(dim=-1, keepdim=True)
        return delta_w

    def get_image_features(self, images, norm=False):
        images = self.psp_preprocess(images)
        encodings, invert_img = self.model(images)
        # encodings = encodings[:, -1:]
        encodings = encodings.view(images.size(0), -1)

        # TODO: different from clip encodings, normalize may be harmful
        if norm:
            encodings /= encodings.clone().norm(dim=-1, keepdim=True)
        return encodings, invert_img
    
    def forward(self, target_imgs, source_imgs):
        target_encodings, _ = self.get_image_features(target_imgs)
        source_encodings, _ = self.get_image_features(source_imgs)

        edit_direction = target_encodings - source_encodings
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        return self.direction_loss(edit_direction, self.target_direction)
        

        