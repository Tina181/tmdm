# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
k_cond=1
label_len=16
k_cond_schedule='linear'

python runner9_NS_transformer.py \
        --model_id "RES_ILI_36_36_kcond_$k_cond_schedule" \
        --model "RES_ILI_36_36_kcond_$k_cond_schedule" \
        --root_path './dataset/illness' \
        --data_path 'national_illness.csv' \
        --seq_len 36 \
        --label_len $label_len \
        --pred_len 36 \
        --train_epochs 200 \
        --gpu 4 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '7' \
        --dec_in '7' \
        --c_out '7' \
        --patience 100 \
        --use_res_diffusion \
        --k_cond $k_cond \
> log/RES_ILI_36_36_large_model_label_len_${label_len}_kcond_schedule_${k_cond_schedule}_k_cond_$k_cond.log 2>&1 &