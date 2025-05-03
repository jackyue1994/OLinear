if [ ! -d "./logs/orthoLinear/main_corr_only/Exchange" ]; then
    mkdir -p ./logs/orthoLinear/main_corr_only/Exchange
fi

model_name=OLinear_C

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(256 128 256 256)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.0 0.0 0.0 0.0)
train_epochs=(1 1 1 1)
bs=(64 32 64 32)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --q_mat_file exchange_rate_${seq_len}_ratio0.7.npy \
      --q_out_mat_file exchange_rate_${pred_len}_ratio0.7.npy \
      --q_channel_file exchange_rate_COV_channel_ratio0.70.npy \
      --model_id Exchange_OLinear_${seq_len}_${pred_len}_corr_only \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size ${bs[i]} \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers 2 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs ${train_epochs[i]} \
      --patience 8 \
      --lradj type1 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 0 \
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_corr_only/Exchange/Exchange_OLinear_${seq_len}_${pred_len}_corr_only.log

done



