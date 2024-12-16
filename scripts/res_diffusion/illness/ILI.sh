# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
python runner9_NS_transformer.py \
        --model_id 'RES_ILI_36_36_large_model' \
        --model 'RES_ILI_36_36_large_model' \
        --root_path './dataset/illness' \
        --data_path 'national_illness.csv' \
        --seq_len 36 \
        --label_len 16 \
        --pred_len 36 \
        --train_epochs 200 \
        --gpu 5 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '7' \
        --dec_in '7' \
        --c_out '7' \
        --patience 500 \
        --use_res_diffusion