# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=0 python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "ViT-B-16+32-invert" \
                --lr 0.002 \
                --frozen_gen_ckpt /home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Fernando_Botero_Painting/ViT-B-16+32-prompt/checkpoint/000300.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 501 \
                --source_class "Fernando Botero Painting" \
                --target_class "photo" \
                --source_type "invert" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 50 \
                --clip_models "ViT-B/16" "ViT-B/32" \
                --clip_model_weights 1.0 1.0 \
                --psp_model_weight 0.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --lambda_within 0 \
                --lambda_across 0.0 \
                # --style_img_dir ../img/mind/1.png \
