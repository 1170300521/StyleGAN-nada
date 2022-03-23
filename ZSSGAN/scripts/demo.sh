# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=1 python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "ViT-B-16+32-global" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_ffhq_encode.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "Super Saiyan" \
                --source_type "zero" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 50 \
                --clip_models "ViT-B/16" "ViT-B/32" \
                --clip_model_weights 1.0 1.0 \
                --psp_model_weight 0.0 \
                --lambda_direction 0.0 \
                --lambda_global 1.0 \
                --lambda_texture 0.0 \
                --lambda_within 0 \
                --lambda_across 0.0 \
                # --style_img_dir ../img/car.jpeg
                # --style_img_dir ../img/mind/1.png \
                # --style_img_dir ../img/aligned/031_971_033_4k_clarisse-debray-lesionsona2_00.png
                
                # --train_gen_ckpt /home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_9/test/checkpoint/original.pt \
