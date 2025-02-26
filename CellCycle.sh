if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TSMixerH" ]; then
    mkdir ./logs/LongForecasting/TSMixerH
fi
seq_len=336
model_name=TSMixerH
dataset=CellCycle
num_channels=6

#Best configuration for ETTm2 and 96 frames horizon
pred_len=96
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9 \
  --num_clusters 2 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 128 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log

pred_len=192
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9 \
  --num_clusters 2 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 128 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

pred_len=336
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9 \
  --num_clusters 2 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 128 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

pred_len=720
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9 \
  --num_clusters 2 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 128 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

