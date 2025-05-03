if [ ! -d "./logs/orthoLinear/main_corr_only/web" ]; then
    mkdir -p ./logs/orthoLinear/main_corr_only/web
fi

model_name=OLinear_C

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)

cuda_ids1=(1 1 1 1)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
epochs=(50 50 50 50)
lradj=(type3 type3 type3 type3)


lrs=(4e-4 5e-4 3e-4 4e-4)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    d_model=${d_models[i]}
    lr=${lrs[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/website/ \
      --data_path website.csv \
      --q_mat_file website_${seq_len}_ratio0.7.npy \
      --q_out_mat_file website_${pred_len}_ratio0.7.npy \
      --q_channel_file website_COV_channel_ratio0.70.npy \
      --model_id website_OLinear_${seq_len}_${pred_len}_${d_model}_${lr}_${lradj[i]}_corr_only \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 4 \
      --dec_in 4 \
      --c_out 4 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_model} \
      --d_ff ${d_model} \
      --batch_size 4 \
      --learning_rate $lr \
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
      2>&1 | tee -a logs/orthoLinear/main_corr_only/web/website_OLinear_${seq_len}_${pred_len}_${d_model}_${lr}_${lradj[i]}_corr_only.log

done