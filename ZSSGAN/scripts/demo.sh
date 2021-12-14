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
CUDA_VISIBLE_DEVICES=0 python train.py --size 512 \
                --batch 2 \
                --n_sample 4 --output_dir ../results/church/pca_latest_16_disney_princess\
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-car-config-f.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "Disney Princess" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1500 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                --crop_for_cars
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
