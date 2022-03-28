# ffhq stylegan2-ffhq-config-f : 1024; cat: 512; dog: 512; church: 256; horse: 256; car: 512, crop_for_cars 
CUDA_VISIBLE_DEVICES=0 python visual.py  \
                --batch 2  --dataset "dog" \
                --n_sample 4 --output_dir "invert" \
                --lr 0.002 \
                --frozen_gen_ckpt ../weights/afhqdog.pt \
                --psp_path ../weights/psp_weights/psp_afhqdog_encode.pt \
                --iter 501 \
                --source_class "photo" \
                --target_class "invert" \
