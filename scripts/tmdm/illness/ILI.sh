# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
python runner9_NS_transformer.py \
        --model_id 'ILI_36_36_model_pro' \
        --model 'ILI_36_36_model_pro' \
        --root_path './dataset/illness' \
        --data_path 'national_illness.csv' \
        --seq_len 36 \
        --label_len 16 \
        --pred_len 36 \
        --train_epochs 10 \
        --gpu 5 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '7' \
        --dec_in '7' \
        --c_out '7' \
        --patience 500