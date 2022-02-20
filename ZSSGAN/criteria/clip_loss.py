import pickle
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import clip
from PIL import Image
from sklearn.decomposition import PCA

from ZSSGAN.utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., \
        lambda_manifold=0., lambda_texture=0., patch_loss_type='mae', \
            direction_loss_type='cosine', clip_model='ViT-B/32', args=None):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.args = args
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        self.patch_text_directions = None

        self.patch_loss     = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.lambda_global    = lambda_global
        self.lambda_patch     = lambda_patch
        self.lambda_direction = lambda_direction
        self.lambda_manifold  = lambda_manifold
        self.lambda_texture   = lambda_texture
        self.alpha = args.alpha

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()
        self.id_loss = DirectionLoss('cosine')

        self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                        preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                        preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor

        self.texture_loss = torch.nn.MSELoss()
        self.pca_components = None
        self.condition = None
        self.pca_threshold = None
        self.clip_threshold = None
        self.clip_mean = None
        self.pca = self.get_pca()

    def get_pca(self):
        orig_sample_path = '../weights/samples.pkl'
        with open(orig_sample_path, 'rb') as f:
            X = pickle.load(f)
            X = np.array(X)
        self.samples = X
        self.clip_mean = torch.from_numpy(np.mean(X, axis=0)).float().to(self.device)
        if self.args.supress == 1:    
            std = np.std(X, axis=0)
            self.clip_threshold = torch.from_numpy(std).float().to(self.device) * self.alpha
            # Get effective dimensions different from origional domain
            self.condition = (std <= (std.mean() * self.args.clip_num_alpha))
            print("The number of style dimensions: ", self.condition.sum())
            # self.condition = torch.from_numpy(self.condition).float().to(self.device)

        # with open("../results/ffhq/clip_orders.pkl",'rb') as f:
        #     orders = pickle.load(f)[0:30]
        #     np.random.shuffle(orders)
        #     self.condition = torch.ones(512)
        #     self.condition[orders] = 0
        #     self.condition = self.condition.to(self.device).unsqueeze(0)
        
        # Define a pca and train it
        pca = PCA(n_components=self.args.pca_dim)
        pca.fit(X)

        # Get the standar deviation of samples and set threshold for each dimension
        threshold = np.sqrt(pca.explained_variance_) * self.alpha
        self.pca_threshold = torch.from_numpy(threshold).float().to(self.device)
        self.pca_components = torch.from_numpy(pca.components_).float().to(self.device)

        # if self.args.enhance:
        #     threshold = threshold * (threshold / (threshold.mean() + 1e-8))
        return pca

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def get_similar_img(self, tgt_vec):
        tgt = tgt_vec[0].cpu().numpy()
        sim = np.dot(self.samples, tgt)
        orders = np.argsort(sim)[::-1]
        print("Orders: {}, Similarities: {}".format(orders[0:20], sim[orders[0:20]]))
        src = self.samples[orders[0:1]]
        src = src * sim[orders[0:1], None]
        src = torch.from_numpy(src).to(tgt_vec.device, dtype=tgt_vec.dtype).mean(axis=0, keepdim=True)
        # src /= src.norm(dim=-1, keepdim=True)
        return src

    def supress_normal_features(self, vec, is_target=False):
        '''
        Supress normal features of the given vector based on original StyleGAN

        Params:
            vec: the vector to be supressed
        '''
        if self.args.supress == 0:
            return vec
        elif self.args.supress == 1:
            if self.condition is None or isinstance(self.condition, np.ndarray):
#                tmp = ((vec[0] - self.clip_mean).abs() / (self.clip_threshold + 1e-8)).cpu().numpy()
#                tmp[tmp > 1] = 1
#                tmp = tmp * (1 - self.condition)
#                self.condition = self.condition + tmp
                self.condition = torch.from_numpy(self.condition).unsqueeze(0).float().to(vec.device)
                print("The number of style and special attrs: ", self.condition.sum())
            #     self.condition = ((vec[0].abs() - self.clip_mean) > self.clip_threshold).unsqueeze(0).float()
            # return vec * self.condition if is_target else vec
            return vec
        elif self.args.supress == 2:
            if self.clip_mean is not None:
                vec = vec - self.clip_mean
            vec_pca = vec @ self.pca_components.t()
            if self.condition is None:
                self.condition = (vec_pca[0].abs() > self.pca_threshold).unsqueeze(0).float()
#                self.condition = torch.ones_like(vec[0]).unsqueeze(0)
#                self.condition[:, self.args.begin:self.args.regular_pca_dim] = 0
            
            return vec_pca * self.condition if is_target else vec_pca
        else:
            raise RuntimeError(f"The choice {self.args.supress} is illegal! Please choose it among 0, 1, 2.")

    def keep_normal_features(self, vec):
        '''
        Keep normal features of the given vector based on original StyleGAN
        '''
        if self.args.supress == 0:
            return vec * 0
        elif self.args.supress == 1:
            return vec * (1 - self.condition)
        elif self.args.supress == 2:
            if self.clip_mean is not None:
                vec = vec - self.clip_mean
            vec_pca = vec @ self.pca_components.t()
#            return vec_pca * (1 - self.condition)
            return vec_pca
        else:
            raise RuntimeError(f"The choice {self.args.supress} is illegal! Please choose it among 0, 1, 2.")

    def get_pca_features(self, vec):
        '''
        Convert CLIP features to PCA features
        '''
        if self.clip_mean is None:
            return vec
        vec = vec - self.clip_mean
        return vec @ self.pca_components.t()

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.clip_mean
        target_features = self.get_text_features(target_class)
        # source_features = self.get_similar_img(target_features)
        self.similar_imgs = self.get_similar_img(target_features)
        
        # Supress normal features and keep special features in the text feature
        target_features = self.supress_normal_features(target_features, is_target=True)
        source_features = self.supress_normal_features(source_features, is_target=True)

        # source_features = 0
        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        # text_direction = target_features.mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def get_raw_img_features(self, imgs: str):
        pre_i = self.clip_preprocess(Image.open(imgs)).unsqueeze(0).to(self.device)
        encoding = self.model.encode_image(pre_i)
        encoding /= encoding.norm(dim=-1, keepdim=True)
        return encoding

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:
        with torch.no_grad():
            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = self.supress_normal_features(target_encoding, is_target=True)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

#            src_encoding = self.get_image_features(source_images)
#            src_encoding = src_encoding.mean(dim=0, keepdim=True)
            # src_encoding = self.get_similar_img(target_encoding)
            # src_encoding = self.get_similar_img(target_encoding)
            self.similar_imgs = self.get_similar_img(target_encoding)
            src_encoding = self.clip_mean
            src_encoding = self.supress_normal_features(src_encoding, is_target=True)
            # src_encoding = 0
            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def compute_corresponding_img2img_direction(self, source_images: list, target_images: list) -> torch.Tensor:
        with torch.no_grad():
            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = self.supress_normal_features(target_encoding, is_target=True)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            source_encodings = []
            for source_img in source_images:
                preprocessed = self.clip_preprocess(Image.open(source_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                source_encodings.append(encoding)
            
            source_encoding = torch.cat(source_encodings, axis=0)
            target_encoding = self.supress_normal_features(target_encoding, is_target=True)
            source_encoding = source_encoding.mean(dim=0, keepdim=True)
            direction = target_encoding - source_encoding
            direction /= direction.norm(dim=-1, keepdim=True)
        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
   
    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        if self.args.use_mean:
            src_encoding = self.clip_mean
        else:
            src_encoding    = self.get_image_features(src_img)
        src_encoding = self.supress_normal_features(src_encoding, is_target=True)

        target_encoding = self.get_image_features(target_img)
        target_encoding = self.supress_normal_features(target_encoding, is_target=True)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        
        return self.direction_loss(edit_direction[:, 0:self.args.divide_line], self.target_direction[:, 0:self.args.divide_line]).mean()

    def pca_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        # if self.args.use_mean:
        #     src_encoding = self.clip_mean
        # else:
        src_encoding    = self.get_image_features(src_img)
        src_encoding = self.get_pca_features(src_encoding)

        target_encoding = self.get_image_features(target_img)
        target_encoding = self.get_pca_features(target_encoding)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        
        return self.direction_loss(edit_direction[:, self.args.divide_line::], self.target_direction[:, self.args.divide_line::]).mean()

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
            
        tokens = clip.tokenize(text).to(self.device)
        image  = self.preprocess(img)

        logits_per_image, _ = self.model(image, tokens)

        return (1. - logits_per_image / 100).mean()

    def adaptive_global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if self.alpha == 0:
            return self.global_clip_loss(img, text)
        text_features = self.get_text_features(text, templates=['{}'])
        img_features = self.get_image_features(img)

        text_features = text_features - self.pca_mean.unsqueeze(0)
        text_features = text_features @ self.pca_cov.t()
        img_features = img_features - self.pca_mean.unsqueeze(0)
        img_features = img_features @ self.pca_cov.t()
        logits_per_img = img_features @ text_features.t()
        return (1. - logits_per_img).mean()
        
    def random_patch_centers(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape

        half_size = size // 2
        patch_centers = np.concatenate([np.random.randint(half_size, width - half_size,  size=(batch_size * num_patches, 1)),
                                        np.random.randint(half_size, height - half_size, size=(batch_size * num_patches, 1))], axis=1)

        return patch_centers

    def generate_patches(self, img: torch.Tensor, patch_centers, size):
        batch_size  = img.shape[0]
        num_patches = len(patch_centers) // batch_size
        half_size   = size // 2

        patches = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):

                center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
                center_y = patch_centers[batch_idx * num_patches + patch_idx][1]

                patch = img[batch_idx:batch_idx+1, :, center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

                patches.append(patch)

        patches = torch.cat(patches, axis=0)

        return patches

    def patch_scores(self, img: torch.Tensor, class_str: str, patch_centers, patch_size: int) -> torch.Tensor:

        parts = self.compose_text_with_templates(class_str, part_templates)    
        tokens = clip.tokenize(parts).to(self.device)
        text_features = self.encode_text(tokens).detach()

        patches        = self.generate_patches(img, patch_centers, patch_size)
        image_features = self.get_image_features(patches)

        similarity = image_features @ text_features.T

        return similarity

    def clip_patch_similarity(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        patch_size = 196 #TODO remove magic number

        patch_centers = self.random_patch_centers(src_img.shape, 4, patch_size) #TODO remove magic number
   
        src_scores    = self.patch_scores(src_img, source_class, patch_centers, patch_size)
        target_scores = self.patch_scores(target_img, target_class, patch_centers, patch_size)

        return self.patch_loss(src_scores, target_scores)

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.patch_text_directions is None:
            src_part_classes = self.compose_text_with_templates(source_class, part_templates)
            target_part_classes = self.compose_text_with_templates(target_class, part_templates)

            parts_classes = list(zip(src_part_classes, target_part_classes))

            self.patch_text_directions = torch.cat([self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

        patch_size = 510 # TODO remove magic numbers

        patch_centers = self.random_patch_centers(src_img.shape, 1, patch_size)

        patches = self.generate_patches(src_img, patch_centers, patch_size)
        src_features = self.get_image_features(patches)

        patches = self.generate_patches(target_img, patch_centers, patch_size)
        target_features = self.get_image_features(patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1), self.patch_text_directions.unsqueeze(0))

        patch_class_scores = cosine_dists * (edit_direction @ self.patch_text_directions.T).softmax(dim=-1)

        return patch_class_scores.mean()

    def cnn_feature_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_features = self.encode_images_with_cnn(src_img)
        target_features = self.encode_images_with_cnn(target_img)

        return self.texture_loss(src_features, target_features)

    def keep_normal_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_encoding    = self.get_image_features(src_img)
        src_encoding = self.keep_normal_features(src_encoding)

        target_encoding = self.get_image_features(target_img)
        target_encoding = self.keep_normal_features(target_encoding)

        return self.id_loss(src_encoding[:, 0:self.args.divide_line], target_encoding[:, 0:self.args.divide_line]).mean()
#        return self.id_loss(src_encoding[:, self.args.begin:self.args.regular_pca_dim], target_encoding[:, self.args.begin:self.args.regular_pca_dim]).mean()

    def remove_similar_loss(self, src_img, target_img):
        target_encoding = self.get_image_features(target_img)
        src_encoding = self.get_image_features(src_img)
        batch, d = target_encoding.shape
        num, d = self.similar_imgs.shape
#        target_encoding = target_encoding.unsqueeze(1).repeat(1, num, 1)
#        src_encoding = src_encoding.unsqueeze(1).repeat(1, num, 1)
#        sim_imgs = self.similar_imgs.unsqueeze(0).repeat(batch, 1, 1)
        #d1 = F.cosine_similarity(src_encoding, sim_imgs, dim=-1) 
        #d2 = F.cosine_similarity(target_encoding, sim_imgs, dim=-1)
        #return F.mse_loss(d1, d2)
        sim_imgs = self.similar_imgs.mean(axis=0, keepdim=True)
        return F.cosine_similarity(target_encoding, sim_imgs, dim=-1)
        # sim_imgs /= sim_imgs.norm(dim=-1, keepdim=True)
        # return 1 - F.mse_loss(target_encoding, sim_imgs)

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, texture_image: torch.Tensor = None):
        clip_loss = 0.0

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])

        if self.lambda_patch:
            clip_loss += self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_direction:
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        # if self.lambda_texture and (texture_image is not None):
        if self.lambda_texture:
            # clip_loss += self.lambda_texture * self.cnn_feature_loss(texture_image, target_img)
            clip_loss += self.lambda_texture * self.cnn_feature_loss(src_img, target_img)
        if self.args.lambda_keep:
            clip_loss += self.args.lambda_keep * self.keep_normal_loss(src_img, target_img)

        if self.args.lambda_pca > 0 and self.args.divide_line < 512:
            clip_loss += self.args.lambda_pca * self.pca_directional_loss(src_img, source_class, target_img, target_class)
        return clip_loss + self.remove_similar_loss(src_img, target_img)
