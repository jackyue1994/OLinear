if [ ! -d "./logs/orthoLinear/main_corr_only/robust/Traffic" ]; then
    mkdir -p ./logs/orthoLinear/main_corr_only/robust/Traffic
fi

model_name=OLinear_C

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --q_mat_file traffic_${seq_len}_ratio0.7.npy \
      --q_out_mat_file traffic_${pred_len}_ratio0.7.npy \
      --q_channel_file traffic_COV_channel_ratio0.70.npy \
      --model_id Traffic_OLinear_${seq_len}_${pred_len}_robust \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate 5e-4 \
      --itr 7 \
      --e_layers 3 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 0 \
      --resume_training 0 \
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 1 \
      --train_epochs 50 \
      --patience 5 \
      --lradj cosine \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_corr_only/robust/Traffic/Traffic_OLinear_${seq_len}_${pred_len}_robust.log

done
