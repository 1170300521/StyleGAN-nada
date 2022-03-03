# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
target_class="Image_4"
output_dir="A_gen_10k-B_500"
psp_alpha=0.4
num_mask_last=10
cuda_id=1

CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class  "$target_class" \
                --alpha 0 \
                --supress 0 \
                --pca_dim 512 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 300 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                --psp_model_weight 0.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --style_img_dir ../img/mind/4.png

CUDA_VISIBLE_DEVICES=$cuda_id python visual.py --size 1024 \
                --batch 2 \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "$target_class" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 150 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 0.0 \
                --psp_model_weight 1.0 \

CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 401 \
                --source_class "photo" \
                --target_class "$target_class" \
                --alpha 0 \
                --supress 0 \
                --pca_dim 512 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1 \
                --psp_model_weight 1 \
                --num_mask_last $num_mask_last \
                --psp_alpha $psp_alpha \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --style_img_dir ../img/mind/4.png \
