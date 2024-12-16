#!/bin/bash
# --data 'ETTh2' \
# multi gpu
# --use_multi_gpu 'True' \
# --devices '7, 6, 5, 4' \
# --gpu '7' \

# singel gpu
# --gpu '7' \

# compare to itransformer
# delete model, add --model,  change dff to 2048 remove  des, d_model dff batch_size learning_rate itr
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_192 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.0005 \
#   --itr 1

# TODO change: gpu enc_in dec_in c_out
python runner9_NS_transformer.py \
        --model_id 'Electricity_96_192' \
        --model 'Electricity_96_192' \
        --root_path './dataset/electricity' \
        --data_path 'electricity.csv' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 200 \
        --gpu 5 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '321' \
        --dec_in '321' \
        --c_out '321' \

