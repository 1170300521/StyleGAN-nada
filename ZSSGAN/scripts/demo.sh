# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=1 python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "regupca_20_32-sup_2-a_0-512" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "Van Goph painting" \
                --alpha 0 \
                --supress 2 \
                --pca_dim 512 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 3000 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --lambda_within 0.5 \
                --lambda_across 0.0 \
                --lambda_keep 0.0 \
                --lambda_pca 0.0 \
                --begin 20 \
                --regular_pca_dim 32 \
                --divide_line 512 \
                --regularize_step 1 \
                # --style_img_dir ../results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/dst_000000.jpg \
                # --style_img_dir_src ../results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/src_000000.jpg
