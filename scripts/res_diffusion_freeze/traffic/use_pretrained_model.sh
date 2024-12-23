# pretrained model is already saved in checkpoints_save1, so we can evaluate it directly.
# --use_multi_gpu 'true' \
# --devices '7, 6, 5, 4, 3, 2, 1' \
python runner9_NS_transformer.py \
        --model_id 'RES_Traffic_96_192_diffusion_step_1000' \
        --model 'RES_Traffic_96_192_diffusion_step_1000' \
        --not_training \
        --pretrained_model_path 'checkpoints/RES_Traffic_96_192_diffusion_step_1000_RES_Traffic_96_192_diffusion_step_1000_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \
        --pretrained_cond_model_path 'checkpoints/RES_Traffic_96_192_diffusion_step_1000_RES_Traffic_96_192_diffusion_step_1000_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/best_cond_model_dir/checkpoint.pth' \
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
        --diffusion_steps '1000' \
        --timesteps '1000' \
        --test_batch_size '8' \
        --use_res_diffusion \
        --save_five_pred_only \
