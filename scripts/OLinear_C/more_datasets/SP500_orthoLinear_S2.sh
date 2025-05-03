if [ ! -d "./logs/orthoLinear/main_corr_only/SP500" ]; then
    mkdir -p ./logs/orthoLinear/main_corr_only/SP500
fi

model_name=OLinear_C

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)


d_models=(256 256 256 256)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)
epochs=(50 50 50 50)
lradj=(type1 type1 type1 type1)
lrs=(2e-4 1e-4 2e-4 2e-4)


for ((i = 0; i < 2; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/SP500/ \
      --data_path SP500.csv \
      --q_mat_file SP500_${seq_len}_ratio0.7.npy \
      --q_out_mat_file SP500_${pred_len}_ratio0.7.npy \
      --q_channel_file SP500_COV_channel_ratio0.70.npy \
      --model_id SP500_OLinear_${seq_len}_${pred_len}_corr_only \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 4 \
      --learning_rate ${lrs[i]} \
      --itr 1 \
      --e_layers 3 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
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
      --checkpoints ./checkpoints \
      2>&1 | tee -a logs/orthoLinear/main_corr_only/SP500/SP500_OLinear_${seq_len}_${pred_len}_corr_only.log

done
