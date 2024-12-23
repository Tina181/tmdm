# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
# add the 3 lines below to use the pretrained model
# --not_training \
# --pretrained_model_path 'checkpoints/ILI_36_36_large_model_ILI_36_36_large_model_custom_ftM_sl36_ll16_pl36_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0' \
# --pretrained_cond_model_path 'checkpoints/ILI_36_36_large_model_ILI_36_36_large_model_custom_ftM_sl36_ll16_pl36_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0' \
python runner9_NS_transformer.py \
        --not_training \
        --pretrained_model_path 'checkpoints/RES_ILI_36_36_large_model_kcond_1.6_RES_ILI_36_36_large_model_kcond_1.6_custom_ftM_sl36_ll24_pl36_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \
        --pretrained_cond_model_path 'checkpoints/RES_ILI_36_36_large_model_kcond_1.6_RES_ILI_36_36_large_model_kcond_1.6_custom_ftM_sl36_ll24_pl36_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/best_cond_model_dir/checkpoint.pth' \
        --model_id 'RES_ILI_36_36_new_test' \
        --model 'RES_ILI_36_36_new_test' \
        --root_path './dataset/illness' \
        --data_path 'national_illness.csv' \
        --seq_len 36 \
        --label_len 24 \
        --pred_len 36 \
        --train_epochs 200 \
        --gpu 5 \
        --features 'M' \
        --n_heads 8 \
        --enc_in '7' \
        --dec_in '7' \
        --c_out '7' \
        --patience 100 \
        --use_res_diffusion \
        --k_cond_schedule 'exp'