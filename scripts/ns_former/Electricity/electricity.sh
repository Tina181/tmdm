model_name=iTransformer
python run_former.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --label_len 48 \
  --gpu 6 \
  --train_epochs 50 \
  --data custom 