model_name=iTransformer

python run_former.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --gpu 3 \
  --train_epochs 50