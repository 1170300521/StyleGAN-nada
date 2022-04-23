# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
num_keep_first=7
cuda_id=1
psp_loss_type="dynamic"
sliding_window_size=30
delta_w_type='mean'

# Training config
psp_alpha=0.7
psp_model_weight=5
lambda_partial=1
lambda_content=0
source_type='mean'
# target_class="david.jpg"
# style_img_dir=../img/Dataset/FFHQ/david.jpg
output_dir="test"

# Dataset info
dataset='ffhq'
frozen_gen_ckpt=../weights/stylegan2-ffhq-config-f.pt
psp_path=../weights/psp_weights/e4e_ffhq_encode.pt

source_class=FFHQ
img_folder=../img/Dataset/FFHQ/

for img in $(ls $img_folder)
do
    CUDA_VISIBLE_DEVICES=$cuda_id python train.py  \
                    --batch 2  --dataset $dataset \
                    --n_sample 4 --output_dir $output_dir \
                    --lr 0.002 \
                    --frozen_gen_ckpt $frozen_gen_ckpt \
                    --psp_path $psp_path \
                    --iter 301 \
                    --source_class $source_class \
                    --target_class "$img" \
                    --source_type $source_type \
                    --alpha 0 \
                    --auto_layer_k 18 \
                    --auto_layer_iters 0 --auto_layer_batch 8 \
                    --output_interval 50 \
                    --mixing 0.0 \
                    --save_interval 100 \
                    --clip_models "ViT-B/32" "ViT-B/16" \
                    --psp_loss_type $psp_loss_type \
                    --clip_model_weights 1.0 1.0 \
                    --psp_model_weight $psp_model_weight \
                    --num_keep_first $num_keep_first \
                    --psp_alpha $psp_alpha \
                    --lambda_direction 1.0 \
                    --lambda_partial $lambda_partial \
                    --lambda_content $lambda_content \
                    --lambda_global 0.0 \
                    --lambda_texture 0.0 \
                    --sliding_window_size $sliding_window_size \
                    --delta_w_type $delta_w_type \
                    --style_img_dir $img_folder$img 
done