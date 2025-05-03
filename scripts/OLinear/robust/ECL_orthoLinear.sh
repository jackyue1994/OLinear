if [ ! -d "./logs/orthoLinear/main_no_varcorr/robust" ]; then
    mkdir -p ./logs/orthoLinear/main_no_varcorr/robust
fi

model_name=OLinear

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)

epochs=(30 50 50 50)
lradj=(cosine cosine cosine cosine)

for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --q_mat_file electricity_${seq_len}_ratio0.7.npy \
      --q_out_mat_file electricity_${pred_len}_ratio0.7.npy \
      --q_channel_file electricity_COV_channel_ratio0.70.npy \
      --model_id ECL_OLinear_${seq_len}_${pred_len}_no_var_corr \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
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
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs ${epochs[i]} \
      --patience 5 \
      --lradj ${lradj[i]} \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --plot_mat_label orthoformer_ECL_vanilla \
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_no_varcorr/robust/ECL_OLinear_${seq_len}_${pred_len}_${lradj[i]}_no_var_corr.log

done
