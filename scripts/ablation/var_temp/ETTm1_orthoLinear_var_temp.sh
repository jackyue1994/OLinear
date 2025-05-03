if [ ! -d "./logs/orthoLinear/ablation/var_temp/ETTm1" ]; then
    mkdir -p ./logs/orthoLinear/ablation/var_temp/ETTm1
fi

model_name=OLinear_ablation_var_temp

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.1 0.1 0.1 0.2)


var_linear_mode=(normal normlin none attn)
var_linear_enable=(1 1 0 0)  # the fourth is meaningless


temp_linear=(1 0 0)
temp_attn_linear=(0 1 0)

strs=(linear normlin none selfattn)


for ((l = 0; l < 4; l++))
do

  for ((m = 0; m < 3; m++))
  do

    # attn is added
    [[ $l -eq 1 && $m -eq 0 || $l -eq 2 && $m -eq 2 || $l -eq 3 && $m -eq 1 ]] && continue

    for ((i = 0; i < 4; i++))
    do

        seq_len=${seq_lens[i]}
        pred_len=${pred_lens[i]}

        export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

        python -u run.py \
          --var_linear_mode ${var_linear_mode[l]} \
          --var_linear_enable ${var_linear_enable[l]} \
          --temp_linear ${temp_linear[m]} \
          --temp_attn_linear ${temp_attn_linear[m]} \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm1.csv \
          --q_mat_file ETTm1_${seq_len}_ratio0.6.npy\
          --q_out_mat_file ETTm1_${pred_len}_ratio0.6.npy\
          --q_channel_file ETTm1_COV_channel_ratio0.60.npy \
          --model_id ETTm1_OLinear_${seq_len}_${pred_len}_${learning_rate[i]}_var_${strs[l]}_temp_${strs[m]} \
          --model $model_name \
          --data ETTm1 \
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
          --train_epochs 30 \
          --patience 8 \
          --lradj type1 \
          --loss_mode L1 \
          --train_ratio 1.0 \
          --dropout ${dropout[i]} \
          --plot_mat_flag 0 \
          --checkpoints ./checkpoints \
          2>&1 | tee -a logs/orthoLinear/ablation/var_temp/ETTm1/ETTm1_OLinear_${seq_len}_${pred_len}_var_${strs[l]}_temp_${strs[m]}.log
    done
  done
done