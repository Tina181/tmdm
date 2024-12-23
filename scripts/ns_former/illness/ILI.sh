# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
model_name=iTransformer
python run_former.py \
        --model_id 'ILI_36_36_model' \
        --model $model_name \
        --root_path './dataset/illness' \
        --data_path 'national_illness.csv' \
        --seq_len 36 \
        --label_len 16 \
        --pred_len 36 \
        --train_epochs 100 \
        --gpu 5 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '7' \
        --dec_in '7' \
        --c_out '7' \
        --patience 50 \
        --d_model 512 \
        --d_ff 2048 \
        --e_layers 2 \
        --d_layers 1 \
        --is_training 1 \
        --data custom \