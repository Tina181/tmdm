# pretrained model is already saved in checkpoints_save1, so we can evaluate it directly.
python runner9_NS_transformer.py \
        --model_id 'RES_Evaluate_traffic_1gpu' \
        --model 'RES_Evaluate_traffic_1gpu' \
        --not_training \
        --pretrained_model_path 'checkpoints_save1/Traffic_96_192_Traffic_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \
        --gpu '7' \
        --root_path './dataset/traffic' \
        --data_path 'traffic.csv' \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '862' \
        --dec_in '862' \
        --c_out '862' \
        --test_batch_size '32' \
        --use_res_diffusion \