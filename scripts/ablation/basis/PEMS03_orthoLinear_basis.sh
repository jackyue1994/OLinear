if [ ! -d "./logs/orthoLinear/ablation/basis/PEMS03" ]; then
    mkdir -p ./logs/orthoLinear/ablation/basis/PEMS03
fi

#model_names=(OLinear_FFT OLinear_wavelet OLinear_wavelet2 OLinear_cheby OLinear_Laguerre OLinear_Legendre)
model_names=(OLinear_FFT OLinear_wavelet_concat OLinear_wavelet2_concat OLinear_cheby OLinear_Laguerre OLinear_Legendre)

seq_lens=(96 96 96 96)
pred_lens=(12 24 48 96)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(0 0 0 0)


for ((l = 1; l < 3; l++))
do
  model_name=${model_names[l]}

  for ((i = 0; i < 4; i++))
  do

      seq_len=${seq_lens[i]}
      pred_len=${pred_lens[i]}

      export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS/ \
        --data_path PEMS03.npz \
        --model_id PEMS03_${model_name}_${seq_len}_${pred_len} \
        --model $model_name \
        --data PEMS \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --embed_size ${embed_size[i]} \
        --d_model ${d_models[i]} \
        --d_ff ${d_models[i]} \
        --batch_size 32 \
        --learning_rate 5e-4 \
        --itr 1 \
        --e_layers 3 \
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
        --save_pdf 1 \
        --train_epochs 40 \
        --patience 10 \
        --lradj cosine \
        --loss_mode L1 \
        --train_ratio 1.0 \
        --dropout 0.0 \
        --plot_mat_flag 0 \
        --checkpoints ./checkpoints \
        2>&1 | tee -a logs/orthoLinear/ablation/basis/PEMS03/PEMS03_${model_name}_${seq_len}_${pred_len}.log

  done
done