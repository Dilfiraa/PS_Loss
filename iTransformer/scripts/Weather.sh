export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

use_ps_loss=$1  # Use MSE loss only 1: Use PS loss
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
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --lradj type1\
  --patience 3\
  --train_epochs 10\
  ---use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_96_${ps_lambda}lambda.log

python -u run.py \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_192_${ps_lambda}lambda.log


python -u run.py \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_336_${ps_lambda}lambda.log


python -u run.py \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_96_720_${ps_lambda}lambda.log

done