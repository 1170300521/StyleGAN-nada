from os import access
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

from ZSSGAN.criteria.clip_loss import DirectionLoss


class RegularizeLoss(torch.nn.Module):
    def __init__(self, device, lambda_within=0.5, lambda_across=1., \
        clip_model='ViT-B/32', dist_loss_type='cosine', args=None):
        super(RegularizeLoss, self).__init__()
        self.device = device
        self.args = args
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
        self.lambda_within = lambda_within
        self.lambda_across = lambda_across
        self.within_dist = DirectionLoss(dist_loss_type)
        self.across_dist = DirectionLoss(dist_loss_type)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
     
    def within_loss(self, src_img_a, src_img_b, tgt_img_a, tgt_img_b):
        v_A = src_img_b - src_img_a
        v_B = tgt_img_b - tgt_img_a

        v_A /= v_A.clone().norm(dim=-1, keepdim=True)
        v_B /= v_B.clone().norm(dim=-1, keepdim=True)

        return self.within_dist(v_A, v_B).mean()
    
    def across_loss(self, src_img_a, src_img_b, tgt_img_a, tgt_img_b):
        v_ref = tgt_img_a - src_img_a
        v_samp = tgt_img_b - src_img_b

        v_ref /= v_ref.clone().norm(dim=-1, keepdim=True)
        v_samp /= v_samp.clone().norm(dim=-1, keepdim=True)

        return self.across_dist(v_ref, v_samp).mean()

    def forward(self, src_imgs, tgt_imgs):
        regular_loss = 0.0
        src_encodings = self.get_image_features(src_imgs)
        tgt_encodings = self.get_image_features(tgt_imgs)

        if self.lambda_across:
            regular_loss += self.lambda_across * self.across_loss(src_encodings[0:1], src_encodings[1:],\
                tgt_encodings[0:1], tgt_encodings[1:])
        if self.lambda_within:
            regular_loss += self.lambda_within * self.within_loss(src_encodings[0:1], src_encodings[1:],\
                tgt_encodings[0:1], tgt_encodings[1:])
        return regular_loss