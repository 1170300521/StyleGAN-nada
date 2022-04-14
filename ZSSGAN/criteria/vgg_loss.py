from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


class Vgg16_pt(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_pt, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # [1, 3; 6, 8; 11, 13, 15; 22, 25, 27, 29]
        # [64 * 2, 128 * 2, 256 * 3, 512 * 4]
        # self.inds = [1,3,6,8,11,13,15,22,29]
        self.inds = [22]
        # mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        # std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        
        # self.register_buffer('mu', mean)
        # self.register_buffer('sig', std)

    def forward_cat(self, x):
        # x = (X-self.mu)/self.sig
        # l2 = [x]
        max_size = 0
        l2 = []
        for i in range(len(self.vgg_layers)):
            x = self.vgg_layers[i].forward(x)  # [:,:,1:-1,1:-1]
            if i in self.inds:
                max_size = max(max_size, x.shape[-1])
                l2.append(x)
        
        up_l2 = []
        for l in l2:
            up_l2.append(F.interpolate(l, size=max_size, mode='bilinear'))

        return torch.cat(up_l2, dim=1)


    def forward(self, X):
        return self.forward_cat(X)

    
class VGGLoss(torch.nn.Module):
    def __init__(self, device, args=None):
        super(VGGLoss, self).__init__()

        self.device = device
        self.args = args
        
        self.model = Vgg16_pt()
        self.model.to(self.device)
        self.vgg_preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            self._convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.target_style = None

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def encode_image(self, img, is_preprocess=True, norm=True):
        if is_preprocess:
            img = self.preprocess(img)
        feat = self.model(img)

        B, C, H, W = feat.shape
        feat = feat.view(B, C, H*W).permute(0, 2, 1)

        if norm:
            feat /= feat.clone().norm(dim=-1, keepdim=True)
        return feat

    def compute_target_features(self, target_images: list):
        with torch.no_grad():
            target_encodings = []
            for target_img in target_images:
                img = self.vgg_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                img_feat = self.encode_image(img, is_preprocess=False)
                target_encodings.append(img_feat)
                target_encoding = torch.cat(target_encodings, axis=0)
                target_encoding = target_encoding.mean(dim=0, keepdim=True)
                target_encoding /= target_encoding.norm(dim=-1, keepdim=True)
        self.target_style = target_encoding
    
    def remd_loss(self, tgt_tokens, style_tokens):
        '''
        REMD Loss referring to style transfer
        '''
        tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)
        style_tokens /= style_tokens.clone().norm(dim=-1, keepdim=True)

        attn_weights = torch.bmm(tgt_tokens, style_tokens.permute(0, 2, 1))

        cost_matrix = 1 - attn_weights
        B, N, M = cost_matrix.shape
        row_values, row_indices = cost_matrix.min(dim=2)
        col_values, col_indices = cost_matrix.min(dim=1)

        row_sum = row_values.mean(dim=1)
        col_sum = col_values.mean(dim=1)

        overall = torch.stack([row_sum, col_sum], dim=1)
        return overall.max(dim=1)[0].mean()

    def forward(self, target_imgs):
        target_encodings = self.encode_image(target_imgs)
        target_style = self.target_style.repeat(target_encodings.shape[0], 1, 1)
        return self.remd_loss(target_encodings, target_style)