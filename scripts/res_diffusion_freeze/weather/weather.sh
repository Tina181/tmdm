# TODO change: model_id, model, gpu enc_in dec_in c_out
python runner9_NS_transformer.py \
        --model_id 'RES_Weather_96_192' \
        --model 'RES_Weather_96_192' \
        --root_path './dataset/weather' \
        --data_path 'weather.csv' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 200 \
        --gpu 2 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '21' \
        --dec_in '21' \
        --c_out '21' \
        --use_res_diffusion \