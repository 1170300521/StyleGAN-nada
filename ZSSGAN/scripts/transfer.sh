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
CUDA_VISIBLE_DEVICES=0 python test.py --size 1024 \
                --batch 2 \
                --n_sample 4 --output_dir ../results/ffhq/sketch \
                --lr 0.002 \
                --frozen_gen_ckpt /home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Anime/1_m-sup_2-a_0-512/checkpoint/000300.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "sketch" \
                --auto_layer_k 18 \
                --auto_layer_iters 1 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/32" "ViT-B/16" \
                --clip_model_weights 1.0 1.0 \
                --sample_truncation 1
