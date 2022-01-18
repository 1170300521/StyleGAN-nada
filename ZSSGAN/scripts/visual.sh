CUDA_VISIBLE_DEVICES=0 python visual.py --size 1024 \
                --batch 2 \
                --n_sample 4 --output_dir ../results/ffhq/ \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --iter 301 \
                --source_class "photo" \
                --target_class "photo" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 150 \
                --clip_models "ViT-B/32" \
                --clip_model_weights 1.0 \
                
                
