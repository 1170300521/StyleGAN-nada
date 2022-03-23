# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
target_class="A sketch with black pencil"
output_dir="ViT-B-16+32-global"
psp_alpha=0.7
num_mask_last=10
cuda_id=1
psp_loss_type="dynamic"
lambda_constrain=0
sliding_window_size=30
delta_w_type='mean'
source_type='mean'


CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_weights/psp_ffhq_encode.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "$target_class" \
                --source_type $source_type \
                --alpha 0 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/32" "ViT-B/16" \
                --psp_loss_type $psp_loss_type \
                --clip_model_weights 1.0 1.0 \
                --psp_model_weight 2 \
                --num_mask_last $num_mask_last \
                --psp_alpha $psp_alpha \
                --lambda_direction 0.0 \
                --lambda_constrain $lambda_constrain \
                --lambda_global 1.0 \
                --lambda_texture 0.0 \
                --sliding_window_size $sliding_window_size \
                --delta_w_type $delta_w_type \
                # --style_img_dir ../img/mind/12.png \