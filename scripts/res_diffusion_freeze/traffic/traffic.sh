# TODO change: model_id, model, gpu enc_in dec_in c_out
# --use_multi_gpu 'true' \
#         --devices '7, 6, 5, 4, 3, 2, 1' \
python runner9_NS_transformer.py \
        --model_id 'RES_Traffic_96_192_diffusion_step_1000' \
        --model 'RES_Traffic_96_192_diffusion_step_1000' \
        --root_path './dataset/traffic' \
        --data_path 'traffic.csv' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --train_epochs 200 \
        --gpu 7 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '862' \
        --dec_in '862' \
        --c_out '862' \
        --timesteps '1000' \
        --diffusion_steps '1000' \
        --test_batch_size '2' \
        --use_res_diffusion \
        --save_five_pred_only \