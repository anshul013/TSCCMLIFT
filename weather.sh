if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TMixerH" ]; then
    mkdir ./logs/LongForecasting/TMixerH
fi
seq_len=336
model_name=TMixerH
dataset=weather
num_channels=21

#Best configuration for weather and 96 frames horizon
pred_len=96
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.3 \
  --num_clusters 4 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 64 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom  \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 192 frames horizon
pred_len=192
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7 \
  --num_clusters 4 \
  --clustering_method 'kmeans' \
  --n_layers 8 \
  --d_ff 32 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom  \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 336 frames horizon
pred_len=336
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7 \
  --num_clusters 4 \
  --clustering_method 'kmeans' \
  --n_layers 2 \
  --d_ff 8 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom  \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 720 frames horizon
pred_len=720
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7 \
  --num_clusters 4 \
  --clustering_method 'kmeans' \
  --n_layers 8 \
  --d_ff 16 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom  \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log  