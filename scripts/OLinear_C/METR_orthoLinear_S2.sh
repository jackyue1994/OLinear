if [ ! -d "./logs/orthoLinear/main_corr_only/METR" ]; then
    mkdir -p ./logs/orthoLinear/main_corr_only/METR
fi

model_name=OLinear_C

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)


train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)
epochs=(50 50 50 50)
lradj=(cosine cosine cosine cosine)
layers=(3 3 3 3)
batch_sizes=(32 32 32 32)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/METR_LA/ \
      --data_path METR_LA.csv \
      --q_mat_file METR-LA_${seq_len}_ratio0.70.npy\
      --q_out_mat_file METR-LA_${pred_len}_ratio0.70.npy\
      --q_channel_file METR_LA_COV_channel_ratio0.70.npy\
      --model_id METR_LA_OLinear_${seq_len}_${pred_len}_corr_only \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 207 \
      --dec_in 207 \
      --c_out 207 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size ${batch_sizes[i]} \
      --learning_rate 2e-4 \
      --itr 1 \
      --e_layers ${layers[i]} \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 17 \
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
      2>&1 | tee -a logs/orthoLinear/main_corr_only/METR/METR_LA_OLinear_${seq_len}_${pred_len}_corr_only.log

done
