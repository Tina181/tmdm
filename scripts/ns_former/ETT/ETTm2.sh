#!/bin/bash
# --data 'ETTh2' \
# multivariate predict
# --use_multi_gpu 'True' \
# --devices '0, 1, 2, 3, 4, 5, 6, 7' \
model_name=iTransformer
python run_former.py \
        --model_id 'ETTm2_96_192' \
        --model $model_name \
        --root_path './dataset/ETT-small/' \
        --data_path 'ETTm2.csv' \
        --data 'ETTm2' \
        --gpu '4' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 50 \
        --gpu 6 \
        --features 'M' \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --is_training 1 \

