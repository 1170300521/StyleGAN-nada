#CUDA_VISIBLE_DEVICES=7 python train.py --size 1024 \
#                --batch 2 \
#                --n_sample 4 --output_dir ../results/ffhq/sketch \
#                --lr 0.002 \
#                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
#                --iter 301 \
#                --source_class "photo" \
#                --target_class "sketch" \
#                --auto_layer_k 18 \
#                --auto_layer_iters 1 --auto_layer_batch 8 \
#                --output_interval 50 \
#                --clip_models "ViT-B/32" "ViT-B/16" \
#                --clip_model_weights 1.0 1.0 \
#                --mixing 0.0 \
#                --save_interval 150 \
# ffhq: 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=1 python train.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "1_m-sup_2-a_0-512-mean_img" \
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
                --save_interval 300 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                --lambda_direction 1.0 \
                --lambda_global 0.0 \
                --lambda_texture 0.0 \
                --lambda_within 0.5 \
                --lambda_across 1 \
                --lambda_keep 0.0 \
                --regularize_step 50000 \
                # --style_img_dir ../results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/dst_000000.jpg \
                # --style_img_dir_src ../results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/src_000000.jpg
#                --use_mean \
#                --style_img_dir ../img/aligned/028_681_722_4k_cameron-mark-zoia_00.png
		#--mixing 0.9
#CUDA_VISIBLE_DEVICES=7 python train.py --size 1024 \
#                --batch 2 \
#                --n_sample 4 --output_dir ../results/ffhq/1sketch-photo \
#                --lr 0.002 \
#                --frozen_gen_ckpt ../results/ffhq/1photo-sketch/checkpoint/000300.pt \
#                --iter 301 \
#                --source_class "sketch" \
#                --target_class "photo" \
#                --auto_layer_k 18 \
#                --auto_layer_iters 1 --auto_layer_batch 8 \
#                --output_interval 50 \
#                --mixing 0.0 \
#                --save_interval 150 \
#                --clip_models "ViT-B/32" "ViT-B/16" \
#                --clip_model_weights 1.0 1.0 \
