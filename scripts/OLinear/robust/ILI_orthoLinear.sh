if [ ! -d "./logs/orthoLinear/main_no_varcorr/robust" ]; then
    mkdir -p ./logs/orthoLinear/main_no_varcorr/robust
fi

model_name=OLinear

pred_lens=(3 6 9 12)
seq_lens=(12 12 12 12)


train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)

epochs=(50 50 50 30)
lradj=(type3 type3 type3 type3)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ILI/ \
      --data_path national_illness.csv \
      --q_mat_file ILI_${seq_len}_ratio0.70.npy \
      --q_out_mat_file ILI_${pred_len}_ratio0.70.npy \
      --q_channel_file ILI_COV_channel_ratio0.70.npy\
      --model_id ILI_OLinear_${seq_len}_${pred_len}_no_var_corr \
      --model $model_name \
      --data custom \
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
      --batch_size 4 \
      --learning_rate 1e-4 \
      --itr 7 \
      --e_layers 2 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 0 \
      --resume_training 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs ${epochs[i]} \
      --patience 10 \
      --lradj ${lradj[i]} \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_no_varcorr/robust/ILI_OLinear_${seq_len}_${pred_len}_no_var_corr.log

done
