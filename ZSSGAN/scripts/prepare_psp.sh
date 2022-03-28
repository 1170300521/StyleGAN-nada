# ffhq stylegan2-ffhq-config-f : 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 

CUDA_VISIBLE_DEVICES=1 python visual.py  \
                --batch 2  --dataset "church" \
                --n_sample 4 --output_dir "preprare_psp" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-church-config-f.pt \
                --psp_path ../weights/psp_weights/e4e_church_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "psp" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" \
                --clip_model_weights 0.0 \
                --psp_model_weight 1.0 \

CUDA_VISIBLE_DEVICES=1 python visual.py  \
                --batch 2  --dataset "cat" \
                --n_sample 4 --output_dir "preprare_psp" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/afhqcat.pt \
                --psp_path ../weights/psp_weights/psp_afhqcat_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "psp" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" \
                --clip_model_weights 0.0 \
                --psp_model_weight 1.0 \

CUDA_VISIBLE_DEVICES=1 python visual.py  \
                --batch 2  --dataset "car" \
                --n_sample 4 --output_dir "preprare_psp" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-car-config-f.pt \
                --psp_path ../weights/psp_weights/e4e_cars_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "psp" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" \
                --clip_model_weights 0.0\
                --psp_model_weight 1.0 \

CUDA_VISIBLE_DEVICES=1 python visual.py  \
                --batch 2  --dataset "dog" \
                --n_sample 4 --output_dir "preprare_psp" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/afhqdog.pt \
                --psp_path ../weights/psp_weights/psp_afhqdog_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "psp" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" \
                --clip_model_weights 0.0 \
                --psp_model_weight 1.0 \

CUDA_VISIBLE_DEVICES=0 python visual.py  \
                --batch 2  --dataset "ffhq" \
                --n_sample 4 --output_dir "preprare_psp" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/stylegan2-ffhq-config-f.pt \
                --psp_path ../weights/psp_weights/e4e_ffhq_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "psp" \
                --source_type "mean" \
                --auto_layer_k 18 \
                --auto_layer_iters 0 --auto_layer_batch 8 \
                --output_interval 50 \
                --mixing 0.0 \
                --save_interval 1000 \
                --clip_models "ViT-B/16" \
                --clip_model_weights 0.0 \
                --psp_model_weight 1.0 \