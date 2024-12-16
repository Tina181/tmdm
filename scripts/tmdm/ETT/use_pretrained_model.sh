# pretrained model is already saved in checkpoints_save1, so we can evaluate it directly.
python runner9_NS_transformer.py \
        --not_training \
        --pretrained_model_path 'checkpoints_save1/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \
        --gpu '0'