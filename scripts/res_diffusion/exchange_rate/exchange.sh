# TODO change: model_id, model, gpu enc_in dec_in c_out
python runner9_NS_transformer.py \
        --model_id 'RES_Exchage_96_192' \
        --model 'RES_Exchage_96_192' \
        --root_path './dataset/exchange_rate' \
        --data_path 'exchange_rate.csv' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 200 \
        --gpu 4 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '8' \
        --dec_in '8' \
        --c_out '8' \
        --use_res_diffusion