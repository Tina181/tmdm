model_name=iTransformer
python run_former.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --gpu 4
  # --d_model 128 \
  # --d_ff 2048 \
  # --itr 1
  # --test_batch_size