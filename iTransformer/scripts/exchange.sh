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
    ps_lambdas=(0.1 0.3 0.5 0.7 1.0)
    loss_name=PS
fi

for ps_lambda in ${ps_lambdas[@]}; do
python -u run.py \
  --is_training 1 \
  --root_path ../datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0001\
  --train_epochs 10\
  --patience 3\
  --lradj type1 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/Exchange_96_96_${ps_lambda}lambda.log


python -u run.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0001\
  --train_epochs 10\
  --patience 3\
  --lradj type1 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/Exchange_96_192_${ps_lambda}lambda.log

python -u run.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0001\
  --train_epochs 10\
  --patience 3\
  --lradj type1 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/Exchange_96_336_${ps_lambda}lambda.log

python -u run.py \
  --is_training 1 \
  --root_path /root/PatchTST_supervised/datasets/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0001\
  --train_epochs 10\
  --patience 3\
  --lradj type1 \
  --use_ps_loss $use_ps_loss\
  --ps_lambda $ps_lambda\
  --patch_len_threshold $patch_len_threshold\
  --itr 1 >logs/${loss_name}/Exchange_96_720_${ps_lambda}lambda.log

done
