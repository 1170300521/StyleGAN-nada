import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

from ZSSGAN.criteria.clip_loss import DirectionLoss
from ZSSGAN.model.psp import pSp


def adjust_sigmoid(x, beta=1):
    return F.sigmoid(beta * x)


class PSPLoss(torch.nn.Module):
    def __init__(self, device, args=None):
        super(PSPLoss, self).__init__()

        self.device = device
        self.args = args

        # Moving Average Coefficient
        self.beta = 0.1
        self.source_mean = self.get_source_mean()
        self.target_mean = self.source_mean
        self.target_set = []
        self.svm_target = []
        self.target_pos = 0

        self.model = pSp(self.args.psp_path, device, output_size=args.size, has_decoder=False)
        self.model.to(device)

        self.psp_preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].]
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.target_direction = self.get_target_direction()
        self.direction_loss = DirectionLoss('cosine')
        self.iter_diff = []
        self.iter_mean = []
        self.svm_C = 1

    def get_source_mean(self):
        source_path = "/home/ybyb/CODE/StyleGAN-nada/results/invert/A_gen_w.npy"
        source_codes = np.load(source_path)
        unmasked_num = 18
        if self.args.num_mask_last > 0:
            unmasked_num = 18 - self.args.num_mask_last
            unmasked_num = max(unmasked_num, 1)
            source_codes = source_codes.reshape((-1, 18, 512))[:, 0:unmasked_num]
        source_codes = torch.from_numpy(source_codes).to(self.device).float().view(-1, unmasked_num*512)
        return source_codes.mean(dim=0, keepdim=True)

    def get_target_direction(self, normalize=True):
        # delta_w_path = os.path.join(self.args.output_dir, 'w_delta.npy')
        delta_w_path = os.path.join(self.args.output_dir, f"{self.args.delta_w_type}_w.npy")

        if os.path.exists(delta_w_path):
            delta_w = np.load(delta_w_path)
        else:
            delta_w = np.ones((18, 512))
        unmasked_num = 18
        if self.args.num_mask_last > 0:
            unmasked_num = 18 - self.args.num_mask_last
            unmasked_num = max(unmasked_num, 1)
            delta_w = delta_w[0: unmasked_num]
        
        delta_w = torch.from_numpy(delta_w).to(self.device).float().flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.args.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        self.cond = torch.zeros(num_channel).to(self.device)
        self.cond[chosen_order] = 1
        self.cond = self.cond.unsqueeze(0)

        print(f"supress_num / overall = {self.cond.sum().item()} / {unmasked_num * 512}")

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
    
    def get_conditional_mask(self):
        if self.args.psp_loss_type == "multi_stage":
            return self.cond, None
        elif self.args.psp_loss_type == "dynamic":
            if self.args.delta_w_type == 'mean':
                delta_w = torch.abs(self.target_mean - self.source_mean)
            else:
                delta_w = self.args.svm_boundary.detach()[:, 0:-1]
            threshold_id = torch.argsort(delta_w[0])[int(self.args.psp_alpha * (18 - self.args.num_mask_last) * 512)]  # batch_size
            bias = delta_w[0, threshold_id].item()
            delta_w = delta_w - bias
            return 1 - adjust_sigmoid(delta_w, beta=50), delta_w
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")

    def update_target_set(self, vec):
        if len(self.target_set) < self.args.sliding_window_size:
            self.target_set.append(vec.mean(0).detach())
        else:
            self.target_set[self.target_pos] = vec.mean(0).detach()
            self.target_pos = (self.target_pos + 1) % self.args.sliding_window_size

    def multi_stage_loss(self, target_encodings, source_encodings):
        if self.cond is not None:
            target_encodings = self.cond * target_encodings
            source_encodings = self.cond * source_encodings
        return F.l1_loss(target_encodings, source_encodings)
        
    def constrained_loss(self, cond):
        return torch.abs(cond.mean(1)-self.args.psp_alpha).mean()

    def update_w_loss(self, target_encodings, source_encodings):
        if self.args.delta_w_type == 'mean':
            # Compute new mean direction of target domain
            # Option 1: Moving Average
            # self.target_mean = self.beta * target_encodings.mean(0, keepdim=True).detach() + \
            #     (1 - self.beta) * self.target_mean

            # Option 2: Sliding Window
            self.update_target_set(target_encodings)
            self.target_mean = torch.stack(self.target_set).mean(0, keepdim=True)
            # Get the editing direction
            delta_w = self.target_mean - self.source_mean
            loss = 0
        elif self.args.delta_w_type == 'svm':
            # See target as pos and source as neg, and only update delta_w
            batch = len(source_encodings)
            source_encodings = -torch.cat([source_encodings.detach(), \
                torch.ones(batch, 1, device=self.device)], dim=-1)
            target_encodings = torch.cat([target_encodings.detach(), \
                torch.ones(batch, 1, device=self.device)], dim=-1)
            samples = torch.cat([target_encodings, source_encodings], dim=0).t()
            w = self.args.svm_boundary
            loss = 0.5 * (w @ w.t()).sum() + self.svm_C * torch.square(F.relu(1 - w @ samples)).sum()
            delta_w = w.detach()[:, 0:-1]
        return loss, delta_w


    def dynamic_loss(self, target_encodings, source_encodings, delta_w):
        # Get the conditional vector to mask special enough channels
        delta_w = torch.abs(delta_w)
        threshold_id = torch.argsort(delta_w[0])[int(self.args.psp_alpha * (18 - self.args.num_mask_last) * 512)]  # batch_size
        bias = delta_w[0, threshold_id].item()
        delta_w = delta_w - bias
        cond = 1 - adjust_sigmoid(delta_w, beta=50)

        # Get masked encodings
        target_encodings = cond * target_encodings
        source_encodings = cond * source_encodings

        # Update the mean direction of target domain and difference
        self.iter_diff.append(F.l1_loss(cond, self.cond).cpu().item())
        self.iter_mean.append(cond.mean().cpu().item())
        
        loss =  F.l1_loss(target_encodings, source_encodings)
        # if self.args.lambda_constrain > 0:
        #     loss += self.constrained_loss(cond)
        return loss

    def forward(self, target_imgs, source_imgs, iters=1):
        target_encodings, _ = self.get_image_features(target_imgs)
        source_encodings, _ = self.get_image_features(source_imgs)

        # Mask w+ codes controlling style and fine details
        if self.args.num_mask_last > 0:
            keep_num = (18 - self.args.num_mask_last) * 512
            target_encodings = target_encodings[:, 0:keep_num]
            source_encodings = source_encodings[:, 0:keep_num]
        
        if self.args.psp_loss_type == "multi_stage":
            # edit_direction = target_encodings - source_encodings
            # theta = (edit_direction.clone() * self.target_direction).sum(dim=-1, keepdim=True)
            # return F.l1_loss(edit_direction, theta * self.target_direction)
            loss = self.multi_stage_loss(target_encodings, source_encodings)
        elif self.args.psp_loss_type == "dynamic":
            loss, delta_w = self.update_w_loss(target_encodings, source_encodings) 
            regular_weight = max(0, \
                    (iters - self.args.sliding_window_size) / (self.args.iter - self.args.sliding_window_size))
            loss += regular_weight * self.dynamic_loss(target_encodings, source_encodings, delta_w=delta_w)
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")
        
        return loss

        
        