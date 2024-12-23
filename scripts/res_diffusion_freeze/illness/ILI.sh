# TODO change: gpu enc_in dec_in c_out
# the sequence length for ILI is set to 36 others: 96
label_len=16
# k_cond_schedule='linear'
# add --freeze_cond_model if want to freeze the cond model
# if only use the pretrained model , use --freeze_cond_model_path

python runner9_NS_transformer.py \
        --model_id "Freeze_RES_ILI_36_36" \
        --model "iTransformer" \
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
        --freeze_cond_model_path 'checkpoints_former/ILI_36_36_model_iTransformer_custom_ftM_sl36_ll16_pl36_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_projection/checkpoint.pth' \
        --freeze_cond_model \