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
dataset=ETTm1
num_channels=7

pred_len=96
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9 \
  --num_clusters 3 \
  --clustering_method 'kmeans' \
  --n_layers 6 \
  --d_ff 16 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $dataset \
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
  --num_clusters 3 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 32 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $dataset \
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
  --num_clusters 3 \
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
  --data $dataset \
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
  --num_clusters 3 \
  --clustering_method 'kmeans' \
  --n_layers 4 \
  --d_ff 16 \
  --in_len $seq_len \
  --out_len $pred_len \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

# #Best configuration for ETTm2 and 96 frames horizon
# pred_len=96
# python3 -u run_longExp.py \
#   --activation 'relu' \
#   --dropout 0.9\
#   --hidden_size 16\
#   --num_clusters 10\
#   --num_blocks 6 \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path $dataset.csv \
#   --model_id $dataset'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in $num_channels \
#   --des 'Exp' \
#   --patience 5 \
#   --train_epochs 100 \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

# #Best configuration for ETTm2 and 192 frames horizon
# pred_len=192
# python3 -u run_longExp.py \
#   --activation 'relu' \
#   --dropout 0.9\
#   --hidden_size 32\
#   --num_clusters 10\
#   --num_blocks 4 \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path $dataset.csv \
#   --model_id $dataset'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in $num_channels \
#   --des 'Exp' \
#   --patience 5 \
#   --train_epochs 100 \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

# #Best configuration for ETTm2 and 336 frames horizon
# pred_len=336
# python3 -u run_longExp.py \
#   --activation 'relu' \
#   --dropout 0.9\
#   --hidden_size 64\
#   --num_clusters 10\
#   --num_blocks 4 \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path $dataset.csv \
#   --model_id $dataset'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in $num_channels \
#   --des 'Exp' \
#   --patience 5 \
#   --train_epochs 100 \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

# #Best configuration for ETTm2 and 720 frames horizon
# pred_len=720
# python3 -u run_longExp.py \
#   --activation 'relu' \
#   --dropout 0.1\
#   --hidden_size 16\
#   --num_clusters 10\
#   --num_blocks 4 \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path $dataset.csv \
#   --model_id $dataset'_'$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in $num_channels \
#   --des 'Exp' \
#   --patience 5 \
#   --train_epochs 100 \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 