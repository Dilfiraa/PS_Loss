export CUDA_VISIBLE_DEVICES=0

seq_len=336
model_name=PatchTST
random_seed=2021
root_path_name=../datasets/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


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


for pred_len in 96 192 336 720; do
for ps_lambda in ${ps_lambdas[@]}; do
    python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.3\
        --fc_dropout 0.3\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 100\
        --revin 1\
        --learning_rate 0.0001\
        --batch_size 128 \
        --use_ps_loss $use_ps_loss\
        --ps_lambda $ps_lambda\
        --patch_len_threshold $patch_len_threshold\
        --itr 1 >logs/${loss_name}/ETTh1_${seq_len}_${pred_len}_${ps_lambda}lambda.log
done
done
