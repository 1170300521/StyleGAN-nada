# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
target_class="026_970_491_4k_jefrey-yonathan-selermun-by-jefreyang_00.png"
output_dir="ViT-B-16+32"
psp_alpha=0.5
num_keep_first=7
cuda_id=1
delta_w_type='mean'
source_type='mean'

CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class  "$target_class" \
                --source_type $source_type \
                --alpha 0 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 300 \
                --clip_models "ViT-B/32" "ViT-B/16"\
                --clip_model_weights 1.0 1.0\
                --psp_model_weight 0.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --style_img_dir ../img/aligned/026_970_491_4k_jefrey-yonathan-selermun-by-jefreyang_00.png

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
                --num_keep_first $num_keep_first \
                --delta_w_type $delta_w_type \

CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir $output_dir \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 401 \
                --source_class "photo" \
                --target_class "$target_class" \
                --source_type $source_type \
                --alpha 0 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/32" "ViT-B/16"\
                --clip_model_weights 1.0 1.0 \
                --psp_model_weight 1 \
                --num_keep_first $num_keep_first \
                --psp_alpha $psp_alpha \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --delta_w_type $delta_w_type \
                --style_img_dir ../img/aligned/026_970_491_4k_jefrey-yonathan-selermun-by-jefreyang_00.png \
