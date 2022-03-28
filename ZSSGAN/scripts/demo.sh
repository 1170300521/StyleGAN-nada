# ffhq stylegan2-ffhq-config-f : 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=0 python train.py  \
                --batch 2  --dataset "dog" \
                --n_sample 4 --output_dir "ViT-B-16-32-mean" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/afhqdog.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "one-shot-clip-dog_2" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" "ViT-B/32" \
                --clip_model_weights 1.0 1.0 \
                --psp_model_weight 0.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --lambda_within 0 \
                --lambda_across 0.0 \
                --style_img_dir ../img/Dataset/one-shot-clip/dog_2.png \
