import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

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
        # delta_w_path = os.path.join(self.args.output_dir, 'w_delta.npy')
        delta_w_path = os.path.join(self.args.output_dir, "small_gen_w.npy")

        if os.path.exists(delta_w_path):
            delta_w = np.load(delta_w_path)
        else:
            delta_w = np.ones((18, 512))
        # delta_w = -np.load('/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/delta.npy').mean(0)
        # delta_w = np.load('/home/ybyb/CODE/StyleGAN-nada/results/invert/tmp.npy')
        # delta_w = delta_w[-1:]
        delta_w = torch.from_numpy(delta_w).to(self.device).float().flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.args.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        self.cond = torch.zeros(num_channel).to(self.device)
        self.cond[chosen_order] = 1
        if self.args.num_mask_last > 0:
            style_mask = torch.cat([torch.ones((18 - self.args.num_mask_last) * 512), \
                torch.zeros(self.args.num_mask_last * 512)], dim=0).to(self.device)
            self.cond = self.cond * style_mask
        self.cond = self.cond.unsqueeze(0)

        print(f"supress_num / overall = {self.cond.sum().item()} / {18 * 512}")

        if normalize:
            delta_w /= delta_w.clone().norm(dim=-1, keepdim=True)
        
        return delta_w.unsqueeze(0)

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

        target_encodings = self.cond * target_encodings
        source_encodings = self.cond * source_encodings
        return F.l1_loss(target_encodings, source_encodings)

        # edit_direction = target_encodings - source_encodings
        # edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        # return self.direction_loss(edit_direction, self.target_direction)

        # num = (18-self.args.num_mask_last) * 512
        # edit_direction = target_encodings - source_encodings
        # edit_direction = edit_direction[:, 0: num]
        # theta = (edit_direction.clone() * self.target_direction[:, 0:num]).sum(dim=-1, keepdim=True)
        # return F.l1_loss(edit_direction, theta * self.target_direction[:, 0:num])
        
        