export CUDA_VISIBLE_DEVICES=0


seq_len=336
model_name=DLinear


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
    ps_lambdas=(0.5 1.0 3.0 5.0 10.0)
    loss_name=PS
fi


for ps_lambda in ${ps_lambdas[@]}; do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ../datasets/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --learning_rate 0.0001\
  --batch_size 16 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_336_96_${ps_lambda}lambda.log


python -u run_longExp.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --learning_rate 0.0001\
  --batch_size 16 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_336_192_${ps_lambda}lambda.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --learning_rate 0.0001\
  --batch_size 16 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_336_336_${ps_lambda}lambda.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --learning_rate 0.0001\
  --batch_size 16 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/weather_336_720_${ps_lambda}lambda.log

done
