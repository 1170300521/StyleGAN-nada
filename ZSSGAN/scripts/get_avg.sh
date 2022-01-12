CUDA_VISIBLE_DEVICES=1 python visual.py  --size 1024\
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "1_m-sup_2-a_0-512" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "Anime" \
                --alpha 0.0 \
                --supress 2 \
                --pca_dim 512 \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1500 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --lambda_within 0.5 \
                --lambda_across 1 \
                --lambda_keep 0.0 \
                --regularize_step 50000 \