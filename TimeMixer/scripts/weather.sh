export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10

use_ps_loss=$1  # 0: Use MSE loss only; 1: Use PS loss
patch_len_threshold=24


if [ ! -d "./logs/MSE/" ];then
    mkdir -p ./logs/MSE/
fi
if [ ! -d "./logs/PS/" ];then
    mkdir -p ./logs/PS/
fi

if [ "$use_ps_loss" -eq 0 ]; then
    ps_lambdas=(0.0)
    loss_name=MSE
else
    ps_lambdas=(1.0 3.0 5.0 10.0)
    loss_name=PS
fi


for ps_lambda in ${ps_lambdas[@]}; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_96_${ps_lambda}lambda.log

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_192_${ps_lambda}lambda.log

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_336_${ps_lambda}lambda.log

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_720_${ps_lambda}lambda.log
done