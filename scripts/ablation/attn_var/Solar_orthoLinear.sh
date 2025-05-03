if [ ! -d "./logs/orthoLinear/ablation/attn_var" ]; then
    mkdir -p ./logs/orthoLinear/ablation/attn_var
fi

model_name=OLinear_attn_var

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)

dropout=(0.0 0.0 0.0 0.0)
learning_rate=(5e-4 5e-4 5e-4 5e-4)

epochs=(40 30 40 30)

lradj=type1

attn_vars=(Transformer Informer Reformer Flowformer Flashformer FLatten LinAttn)  #Mamba

for ((l = 0; l < 7; l++))
do

  attn_var=${attn_vars[l]}

  for ((i = 0; i < 4; i++))
  do

      seq_len=${seq_lens[i]}
      pred_len=${pred_lens[i]}

      export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

      python -u run.py \
        --attn_type ${attn_var} \
        --is_training 1 \
        --root_path ./dataset/Solar/ \
        --data_path solar_AL.txt \
        --q_mat_file solar_AL_${seq_len}_ratio0.7.npy \
        --q_out_mat_file solar_AL_${pred_len}_ratio0.7.npy \
        --q_channel_file Solar_AL_COV_channel_ratio0.70.npy \
        --model_id orthoLinear_Solar_${seq_len}_${pred_len}_${attn_var}_no_var_corr \
        --model $model_name \
        --data Solar \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \
        --embed_size ${embed_size[i]} \
        --d_model ${d_models[i]} \
        --d_ff ${d_models[i]} \
        --batch_size 32 \
        --learning_rate ${learning_rate[i]} \
        --itr 1 \
        --e_layers 2 \
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
        --lradj ${lradj} \
        --loss_mode L1 \
        --train_ratio 1.0 \
        --dropout ${dropout[i]} \
        --plot_mat_flag 0 \
        --linear_attention 0 \
        --checkpoints ./checkpoints \
        2>&1 | tee -a logs/orthoLinear/ablation/attn_var/Solar_OLinear_${seq_len}_${pred_len}_${attn_var}_no_var_corr.log

  done
done