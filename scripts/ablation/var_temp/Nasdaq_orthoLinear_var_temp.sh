if [ ! -d "./logs/orthoLinear/ablation/var_temp/Nasdaq" ]; then
    mkdir -p ./logs/orthoLinear/ablation/var_temp/Nasdaq
fi

model_name=OLinear_ablation_var_temp

pred_lens=(3 6 9 12)
seq_lens=(12 12 12 12)


train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)
epochs=(50 50 50 50)
lradj=(type1 type1 type1 type1)


var_linear_mode=(normal normlin none)
var_linear_enable=(1 1 0)


temp_linear=(1 0 0)
temp_attn_linear=(0 1 0)

strs=(linear normlin none)


for ((l = 0; l < 3; l++))
do

  for ((m = 0; m < 3; m++))
  do

    [[ $l -eq 1 && $m -eq 0 || $l -eq 2 && $m -eq 2 ]] && continue

    for ((i = 0; i < 4; i++))
    do

        seq_len=${seq_lens[i]}
        pred_len=${pred_lens[i]}

        export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

        python -u run.py \
          --var_linear_mode ${var_linear_mode[l]} \
          --temp_linear ${temp_linear[m]} \
          --temp_attn_linear ${temp_attn_linear[m]} \
          --var_linear_enable ${var_linear_enable[l]} \
          --is_training 1 \
          --root_path ./dataset/nasdaq/ \
          --data_path nasdaq.csv \
          --q_mat_file nasdaq_${seq_len}_ratio0.70.npy \
          --q_out_mat_file nasdaq_${pred_len}_ratio0.70.npy \
          --q_channel_file nasdaq_COV_channel_ratio0.70.npy \
          --model_id Nasdaq_OLinear_${seq_len}_${pred_len}_var_${strs[l]}_temp_${strs[m]} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --enc_in 12 \
          --dec_in 12 \
          --c_out 12 \
          --des 'Exp' \
          --embed_size ${embed_size[i]} \
          --d_model ${d_models[i]} \
          --d_ff ${d_models[i]} \
          --batch_size 4 \
          --learning_rate 1e-4 \
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
          2>&1 | tee -a logs/orthoLinear/ablation/var_temp/Nasdaq/Nasdaq_OLinear_${seq_len}_${pred_len}_var_${strs[l]}_temp_${strs[m]}.log

    done
  done
done