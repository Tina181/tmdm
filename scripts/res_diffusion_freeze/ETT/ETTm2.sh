#!/bin/bash
# --data 'ETTh2' \
# multivariate predict
# --use_multi_gpu 'True' \
# --devices '0, 1, 2, 3, 4, 5, 6, 7' \
python runner9_NS_transformer.py \
        --model_id 'RES_ETTm2_96_192' \
        --model 'RES_ETTm2_96_192' \
        --root_path './dataset/ETT-small/' \
        --data_path 'ETTm2.csv' \
        --data 'ETTm2' \
        --gpu '5' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 200 \
        --gpu 6 \
        --features 'M' \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --use_res_diffusion \
