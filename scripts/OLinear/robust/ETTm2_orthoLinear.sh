if [ ! -d "./logs/orthoLinear/main_no_varcorr/robust" ]; then
    mkdir -p ./logs/orthoLinear/main_no_varcorr/robust
fi

model_name=OLinear

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 256 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.2 0.2 0.2 0.3)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --q_mat_file ETTm2_${seq_len}_ratio0.6.npy\
      --q_out_mat_file ETTm2_${pred_len}_ratio0.6.npy\
      --q_channel_file ETTm2_COV_channel_ratio0.60.npy \
      --model_id ETTm2_OLinear_${seq_len}_${pred_len}_robust_no_var_corr \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate ${learning_rate[i]} \
      --itr 7 \
      --e_layers 1 \
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
      --save_pdf 0 \
      --train_epochs 30 \
      --patience 8 \
      --lradj type1 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 0 \
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_no_varcorr/robust/orthoLinear_ETTm2_${seq_len}_${pred_len}_robust_no_var_corr.log
done

