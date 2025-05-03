import argparse
import math
import os
import sys
import time
import shutil
import torch
from experiments.exp_forecast import Exp_Forecast
import random
import numpy as np
from datetime import datetime

from utils.tools import send_email, find_most_recently_modified_subfolder


def get_setting(args_, iter_=0):
    if args_.task_name == 'forecasting':
        setting_ = '{}_{}_{}_{}_sl-{}_pl-{}_var-{}_dm-{}_stages-{}_{}P{}i_{}P{}i_dec-{}_des-{}_{}'.format(
            args_.task_name,
            args_.model_id,
            args_.model,
            args_.data,
            args_.seq_len,
            args_.pred_len,
            args_.enc_in,
            args_.d_model,
            args_.git_multi_stage,
            args_.Patch_layer_num,
            args_.e_layers,
            args_.Patch_layer_num2,
            args_.second_e_layers,
            args_.decoder_cat_num,
            args_.des,
            iter_)
    else:
        setting_ = '{}_{}_{}_{}_sl-{}_mr-{}_var-{}_dm-{}_stages-{}_{}P{}i_{}P{}i_dec-{}_des-{}_{}'.format(
            args_.task_name,
            args_.model_id,
            args_.model,
            args_.data,
            args_.seq_len,
            args_.mask_rate,
            args_.enc_in,
            args_.d_model,
            args_.git_multi_stage,
            args_.Patch_layer_num,
            args_.e_layers,
            args_.Patch_layer_num2,
            args_.second_e_layers,
            args_.decoder_cat_num,
            args_.des,
            iter_)
    return setting_[:255]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GiTransformer')

    # orthoLinear
    parser.add_argument('--CovMatTrans', type=str, default='softmax', help='CovMatTrans for ablations')
    parser.add_argument('--WeightTrans', type=str, default='softplus', help='WeightTrans for ablations')
    parser.add_argument('--NormSet', type=str, default='L1', help='NormSet for ablations')
    parser.add_argument('--pre_lin', type=int, default=1, help='pre_linear for orthoLinear')
    parser.add_argument('--post_lin', type=int, default=1, help='post_linear for orthoLinear')
    parser.add_argument('--var_linear_mode', type=str, default='attn_linear', help='var_linear_mode for orthoLinear')
    parser.add_argument('--temp_linear', type=int, default=1, help='temp_linear for orthoLinear')
    parser.add_argument('--temp_attn_linear', type=int, default=0, help='temp_attn_linear for orthoLinear')
    parser.add_argument('--var_linear_enable', type=int, default=1, help='var_linear_enable for orthoLinear')
    parser.add_argument('--ortho_enc', type=int, default=1, help='ortho_enc for orthoLinear')
    parser.add_argument('--ortho_dec', type=int, default=1, help='ortho_dec for orthoLinear')
    parser.add_argument('--ortho_n_heads', type=int, default=2, help='ortho_n_heads for orthoLinear')

    parser.add_argument('--PatchTST_linear', type=int, default=0, help='PatchTST_linear for PatchTST')
    parser.add_argument('--iTrans_linear', type=int, default=0, help='iTrans_linear for iTrans')
    parser.add_argument('--Leddam_attnLinear', type=int, default=0, help='Leddam_attnLinear for Leddam')

    parser.add_argument('--iTrans_ortho_trans', type=int, default=0, help='iTrans_ortho_trans for iTrans')
    parser.add_argument('--PatchTST_ortho_trans', type=int, default=0, help='PatchTST_ortho_trans for PatchTST')
    parser.add_argument('--DLinear_ortho_trans', type=int, default=0, help='PatchTST_ortho_trans for PatchTST')

    # used for ablations (model=orthoLinear_ablation_lin_design)
    parser.add_argument('--onlyconv', type=int, default=0, help='onlyconv for orthoLinear')

    # part train
    parser.add_argument('--train_part_first', type=float, default=0.0, help='train_part_first for dataLoader')

    parser.add_argument('--attn_type', type=str, default='Enhanced', help='attention types')

    # save weight
    parser.add_argument('--save_linear_weight', type=int, default=0, help='save_linear_weight for orthoLinear')
    parser.add_argument('--save_linear_weight_path', type=str, default='weight_npy',
                        help='save_linear_weight_path for orthoLinear')
    parser.add_argument('--save_linear_weight_tag', type=str, default='tag',
                        help='save_linear_weight_tag for orthoLinear')

    # orthoformer
    parser.add_argument('--q_mat_file', type=str, default=None, help='q_mat_file npy file')
    parser.add_argument('--c_mat_file', type=str, default=None, help='c_mat_file npy file')
    parser.add_argument('--q_channel_file', type=str, default=None, help='q_channel_file npy file')
    parser.add_argument('--Q_MAT_file', type=str, default=None, help='Q_MAT_file npy file')
    parser.add_argument('--q_out_mat_file', type=str, default=None, help='q_out_mat_file npy file')
    parser.add_argument('--c_out_mat_file', type=str, default=None, help='c_out_mat_file npy file')
    parser.add_argument('--Q_OUT_MAT_file', type=str, default=None, help='Q_OUT_MAT_file npy file')
    parser.add_argument('--Q_chan_indep', type=int, default=0, help='Q_channel_independence')
    parser.add_argument('--Q_loss', type=int, default=0, help='use Q_mat in loss function')
    parser.add_argument('--FFT_loss', type=int, default=0, help='use FFT_loss in loss function')
    parser.add_argument('--Q_loss_alpha', type=float, default=0.5, help='Q_loss_alpha')
    parser.add_argument('--dim_reduce_ratio', type=float, default=1.0, help='dim_reduce_ratio')

    # timeMixer++
    parser.add_argument('--channel_mixing', type=int, default=1,
                        help='0: channel mixing 1: whether to use channel_mixing')

    # mamba
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')

    # leddam
    parser.add_argument('--pe_type', type=str, default='no', help='position embedding type')

    # card
    parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.3, help='head dropout')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--use_statistic', type=int, default=0, help='use_statistic')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum')
    parser.add_argument('--merge_size', type=int, default=2)

    # freformer
    parser.add_argument('--embed_size', type=int, default=8, help='embed_size')
    parser.add_argument('--plot_mat_flag', type=int, default=0, help='plot_mat_flag')
    parser.add_argument('--plot_grad_flag', type=int, default=0, help='plot_grad_flag')
    parser.add_argument('--time_branch', type=int, default=0, help='time_branch')
    parser.add_argument('--CKA_flag', type=int, default=0, help='CKA_flag')
    parser.add_argument('--checkpoint_check', type=int, default=0, help='if checkpoint exists, skip training')
    parser.add_argument('--attn_enhance', type=int, default=1, help='attn_enhance')
    parser.add_argument('--attn_softmax_flag', type=int, default=1, help='attn_softmax_flag')
    parser.add_argument('--attn_weight_plus', type=int, default=0, help='attn_weight_plus')
    parser.add_argument('--attn_outside_softmax', type=int, default=0, help='attn_outside_softmax')
    parser.add_argument('--plot_mat_label', type=str, default='dataset', help='plot_mat_label; only '
                                                                              'effective when plot_mat_flag enabled')

    # alternative training
    parser.add_argument('--input_include', type=int, default=0, help='input_include')
    parser.add_argument('--alpha_learnable', type=int, default=0, help='alpha_learnable')
    parser.add_argument('--use_cross', type=int, default=0, help='use_cross')
    parser.add_argument('--base_forecaster', type=str, default='HierPatchTST', help='base_forecaster',
                        choices=['PatchTST', 'HierPatchTST', 'iTransformer', 'TimeMixer', 'DLinear', 'PiT'])
    parser.add_argument('--refine_forecaster', type=str, default='HierPatchTST', help='refine_forecaster',
                        choices=['HierPatchTST', 'iTransformer', 'PiT', 'None'])
    # alternative training
    parser.add_argument('--alter_opt', type=int, default=0, help='alter_opt')
    parser.add_argument('--epoch_for_base', type=int, default=5, help='epoch_for_base')

    # timemixer timeMixer++
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')

    parser.add_argument('--train_ratio', type=float, default=1.0, help='percentage of training set')

    parser.add_argument('--save_pdf', type=int, default=0, help='save_pdf')
    parser.add_argument('--copy_file', type=int, default=0, help='copy_file')

    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--m4_result_path_str', type=str, default='', help='label')
    parser.add_argument('--m4_result_path', type=str, default='./m4_results/',
                        help='specify m4_result_path; effective for test_mode being 3')

    parser.add_argument('--send_mail', type=int, default=0, help='send mail after training completes.')

    # long-short term attention
    parser.add_argument('--win_size_list', type=int, nargs='+', default=[5], help='win_size_list for imputation')
    parser.add_argument('--multi_period_list', type=int, nargs='+', default=[5], help='win_size_list for imputation')

    # imputation
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--non_mask_lamda', type=float, default=0.5, help='non_mask_lamda')

    parser.add_argument('--token_weight', type=int, default=1, help='token_weight')
    parser.add_argument('--token_weight_softmax', type=int, default=0,
                        help='1: token_weight_softmax; 0: pow; -1: F.normalize')
    # ij_mat_flag does not work for cross-variate attention
    parser.add_argument('--ij_mat_flag', type=int, default=1, help='ij_mat_flag for imputation')
    parser.add_argument('--ij_mat_tau_initial', type=float, default=3.0, help='ij_mat_tau_initial')
    parser.add_argument('--token_weight_tau_initial', type=float, default=5.0,
                        help='used for computing token weights (time) when token_weight_softmax is enabled')
    parser.add_argument('--token_weight_tau_initial_chan', type=float, default=5.0,
                        help='used for computing token weights (variates) when token_weight_softmax is enabled')
    parser.add_argument('--token_weight_tau2_initial', type=float, default=5.0,
                        help='used for computing token weights (time) when token_weight_softmax is disabled')
    parser.add_argument('--token_weight_tau2_initial_chan', type=float, default=5.0,
                        help='used for computing token weights (variates) when token_weight_softmax is disabled')
    parser.add_argument('--eval_flag', type=int, default=1, help='evaluation flag for imputation')
    parser.add_argument('--eval_lamda', type=float, default=1.5, help='eval_lamda for imputation loss')
    parser.add_argument('--eval_alpha', type=float, default=-0.0, help='alpha in loss function')

    parser.add_argument('--card_flag', type=int, default=0, help='use card instead of TimesNet')
    parser.add_argument('--timesnet_flag', type=int, default=0, help='timesnet for first stage')
    parser.add_argument('--timesnet_layers', type=int, default=0, help='timesnet_flag for more stages')
    parser.add_argument('--no_feat_mix', type=int, default=0, help='no_feat_mix between timesnet and pit')
    parser.add_argument('--top_k', type=int, default=2, help='periods for TimesNet and myConv')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception in TimesNet')

    # conv for imputation
    parser.add_argument('--imp_first_conv', type=int, default=1, help='use conv/Swin for imputation')
    parser.add_argument('--imp_more_conv', type=int, default=1, help='use conv/Swin for imputation')
    parser.add_argument('--SwinTransList', type=int, default=0, help='SwinTransList for imputation')
    parser.add_argument('--SwinTransBlockNum', type=int, default=3, help='SwinBlockNum for imputation')
    parser.add_argument('--fft_flag', type=int, default=1, help='fft in myConv/Swin')
    parser.add_argument('--Swin_output', type=int, default=0, help='independent swin output flag')
    parser.add_argument('--Conv_block_num', type=int, default=1, help='Conv_block_num')
    parser.add_argument('--Swin_before_patch', type=int, default=0, help='Swin_before_patch')
    parser.add_argument('--Swin_after_patch', type=int, default=0, help='Swin_after_patch')
    parser.add_argument('--Swin_after_iTrans', type=int, default=0, help='Swin_after_iTrans')
    parser.add_argument('--Swin_Shuffle_Num', type=int, default=0, help='Swin_Shuffle_Num; useless. do not use')
    parser.add_argument('--Shuffle_All', type=int, default=0, help='Swin_Shuffle_Num; useless. do not use')
    parser.add_argument('--Swin_attn_mask', type=int, default=0, help='whether use attention mask in swin transformer')
    parser.add_argument('--Swin_win_size', type=int, default=7, help='Swin_win_size')
    parser.add_argument('--Swin_all_attn_flag', type=int, default=0, help='Swin_all_attn_flag')
    parser.add_argument('--SDA_bidirect', type=int, default=0, help='SDA_bidirect')
    parser.add_argument('--SDA_rel_pos_pow_mode', type=int, default=1, help='SDA_rel_pos_pow_mode')
    parser.add_argument('--SDA_use_or_not', type=int, default=1, help='SDA_use_or_not')
    parser.add_argument('--Neighbor_fix', type=int, default=0, help='Neighbor_fix')
    # task_name
    parser.add_argument('--task_name', type=str, default='forecasting',
                        help='task_name', choices=['forecasting', 'long_term_forecast',
                                                   'imputation', 'short_term_forecast'])

    # C-PiT ablation
    parser.add_argument('--input_concat', type=int, default=1, help='ablation study for C-PiT')

    # Patching
    parser.add_argument('--temp_stride', type=int, default=8, help='temp_stride for temporal patching')
    parser.add_argument('--temp_patch_len', type=int, default=16, help='temp_patch_len for patching')
    parser.add_argument('--temp_patch_len2', type=int, default=16, help='temp_patch_len2')
    parser.add_argument('--temp_stride2', type=int, default=8, help='temp_stride2')
    parser.add_argument('--Patch_CI', type=int, default=1, help='use channel independence or not')

    # C-PiT
    parser.add_argument('--git_multi_stage', type=int, default=4, help='git_multi_stage')
    parser.add_argument('--lamda1', type=float, default=1.0, help='lamda1 in loss function')
    parser.add_argument('--lamda1_delta', type=float, default=0.0, help='lamda1 in loss function')
    parser.add_argument('--mapping_fun', type=str, default='softmax_learn', help='mapping_fun',
                        choices=['softmax_learn', 'softmax_q_k', 'x_3', 'relu', 'elu_plus_1', 'agent'])
    parser.add_argument('--flow_attn', type=int, default=0, help='flow attention')
    parser.add_argument('--flash_attn', type=int, default=0, help='flash attention')

    # usually do not need to change second_e_layers
    parser.add_argument('--second_e_layers', type=int, default=1, help='second_e_layers')
    parser.add_argument('--attn_lookback', type=int, default=2, help='attn_lookback, not larger than git_multi_stage')
    parser.add_argument('--find_best', action='store_true', default=False, help='parameter search')
    parser.add_argument('--find_best_num', type=int, default=3, help='parameter search')
    parser.add_argument('--use_revin', type=int, default=1, help='use revin to norm and de-norm')
    parser.add_argument('--linear_attention', type=int, default=0, help='linear_attention')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha in loss function')
    parser.add_argument('--loss_mode', type=str, default='L1', help='loss_mode',
                        choices=['L1', 'L2', 'L1L2', 'MAPE', 'MASE', 'SMAPE'])
    parser.add_argument('--attn_loss_mode', type=str, default='L1', help='attn_loss_mode, not used',
                        choices=['L1', 'L2', 'KL', 'JSD', 'Wasserstein'])
    parser.add_argument('--decoder_cat_num', type=int, default=2, help='decoder_cat_num')
    parser.add_argument('--seq_inter', type=int, default=1, help='check mse/mae in immediate stages')
    parser.add_argument('--Patch_layer_num', type=int, default=2, help='Patch_layer_num')
    parser.add_argument('--Patch_layer_num2', type=int, default=1, help='Patch_layer_num2')
    parser.add_argument('--lossfun_alpha', type=float, default=0.0, help='alpha in loss function')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test_batch_size')
    parser.add_argument('--no_batchsize_search', type=int, default=1, help='no_batchsize_search')
    parser.add_argument('--dp_rank', type=int, default=0, help='dp_rank for dynamic projection')
    parser.add_argument('--fix_seed', type=int, default=1, help='fix_seed')
    parser.add_argument('--grad_clip', type=int, default=0, help='fix_seed')
    parser.add_argument('--max_norm', type=float, default=1e6, help='dp_rank for dynamic projection')

    parser.add_argument('--PatchTST_hier', type=int, default=1, help='PatchTST_hier')
    parser.add_argument('--patch_ln', type=int, default=0, help='gpu')
    parser.add_argument('--test_mode', type=int, default=0, help='gpu')
    parser.add_argument('--patch_multi', type=int, default=1,
                        help='patch_multi for memory saving, only for large pred_len')
    parser.add_argument('--save_every_epoch', type=int, default=0, help='save_every_epoch')
    parser.add_argument('--model_stats_mode', type=int, default=0, help='model_stats_flag')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                             'predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min '
                             'or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  # no longer needed in
    # inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False,
                        help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # monte carlo experiments
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        choices=['type1', 'type2', 'type3', 'card', 'cosine', 'constant', 'TST'],
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # continue training
    parser.add_argument('--resume_training', type=int, default=0, help='resume training')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='whether to use channel_independence mechanism')
    # does not use the real data, but scaled data
    parser.add_argument('--inverse', type=int, default=0, help='inverse output data')
    # class_strategy seems not used
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./dataset/electricity/',
                        help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=int, default=0,
                        help='whether to use efficient_training ')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0,
                        help='the start index of variates for partial training, '
                             'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('args.use_gpu:', args.use_gpu)

    if args.train_ratio < 1:
        print(f'Only {args.train_ratio:.2%} of training set is used.')
    else:
        print(f'All training set is used.')

    # check
    if args.resume_training > 0 >= args.resume_epoch:
        confirm_again = input('args.resume_training > 0 >= args.resume_epoch. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            print('program exists.')
            sys.exit()

    # shuffle is useless!
    if args.Swin_Shuffle_Num > 0:
        print('Warning: args.Swin_Shuffle_Num > 0. Shuffle usually does not work well. ')

    if args.Conv_block_num > 1:
        confirm_again = input('args.Conv_block_num > 1. Conv_block_num should be 1. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            print('program exists.')
            sys.exit()

    if not args.use_gpu:
        confirm_again = input('No using gpu. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            print('program exists.')
            sys.exit()
    if (args.itr > 1 and args.fix_seed) or (args.itr == 1 and args.fix_seed == 0):
        confirm_again = input('Please check args.itr and args.fix_seed. They seem irrational. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            print('program exists.')
            sys.exit()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'imputation':
        # batch_y is useless for imputation
        args.pred_len = args.seq_len

        # check timesnet_layers
        if args.timesnet_layers > args.git_multi_stage + 1:
            print('timesnet_layers are adjusted to git_multi_stage+1...')
            args.timesnet_layers = args.git_multi_stage + 1

        if args.timesnet_layers >= 1 and not args.timesnet_flag:
            print('timesnet_layers overwrites timesnet_flag...')
            args.timesnet_flag = 1

    print('Args in experiment:')

    Exp = Exp_Forecast
    if args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
        args.m4_folder = os.path.join('./m4_results', args.model + '_' + args.m4_result_path_str)
        #  + '_' + datetime.now().strftime('%y%m%d_%H%M%S')
    elif args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation

    # args.model_id
    model_id_ori = args.model_id
    args.model_id_ori = model_id_ori
    args.model_id = model_id_ori + '_' + datetime.now().strftime('%y%m%d_%H%M%S')

    log_txt = 'log.txt'
    best_log_txt = 'best_log.txt'
    best_log_dataset_path = 'best_results'
    if not os.path.exists(best_log_dataset_path):
        os.makedirs(best_log_dataset_path)
    best_log_dataset_txt = os.path.join(best_log_dataset_path, model_id_ori + '.txt')

    test_batch_size_list = [args.test_batch_size]

    global_time0 = time.time()

    if args.fix_seed:
        fix_seed = 2023  # 2023  # if args.task_name == 'forecasting' else 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    setting_zero = get_setting(args, 0)
    folder_path = os.path.join('results', setting_zero)
    args.folder_path = folder_path

    best_mse, best_mae = math.inf, math.inf
    time_vec = []
    if args.is_training and not args.model_stats_mode and not args.save_linear_weight:
        lamda1_ori = args.lamda1
        best_lamda1 = lamda1_ori
        test_batch_size_ori = args.test_batch_size
        best_ii = 0

        # test_batch_size_list
        if test_batch_size_ori not in test_batch_size_list:
            test_batch_size_list.append(test_batch_size_ori)

        # copy file
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if args.copy_file:
            shutil.copytree('./model', os.path.join(folder_path, 'model'))
            shutil.copytree('./layers', os.path.join(folder_path, 'layers'))
            shutil.copytree('./utils', os.path.join(folder_path, 'utils'))
            shutil.copytree('./experiments', os.path.join(folder_path, 'experiments'))
            shutil.copytree('./scripts', os.path.join(folder_path, 'scripts'))
            shutil.copytree('./data_provider', os.path.join(folder_path, 'data_provider'))
            print('Some python files have been copied...')

        # check
        if args.checkpoint_check and not args.resume_training and args.test_mode == 0:
            full_folder, new_setting = find_most_recently_modified_subfolder('./checkpoints/',
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=args.model_id_ori)
            if full_folder is not None:
                print(f'{args.model_id_ori} checkpoints already exist.')
                sys.exit()

        idx = 0
        for ii in range(args.itr):
            # setting record of experiments
            # setting = get_setting(args, ii)

            if args.find_best and args.find_best_num > 1 and args.lamda1_delta > 0:
                lamda1_delta = args.lamda1_delta
                if lamda1_delta > 0:
                    lamda1_list = np.linspace(max(args.lamda1 - lamda1_delta, 0.1), args.lamda1 + lamda1_delta,
                                              args.find_best_num)
                else:
                    lamda1_list = [args.lamda1]
            else:
                lamda1_delta = 0
                lamda1_list = [args.lamda1]

            itr_count = args.itr * len(lamda1_list)
            best_string = ''
            exp = None  # to make pycharm happy
            for lamda1 in lamda1_list:
                # idx and time
                idx = idx + 1
                time_now = time.time()

                args.lamda1 = lamda1
                args.model_id = model_id_ori + '_' + datetime.now().strftime('%y%m%d_%H%M%S')

                setting = get_setting(args, ii) if ii > 0 else setting_zero

                if ii > 0:
                    args.folder_path = os.path.join('results', setting)

                exp = Exp(args)  # set experiments

                if args.test_mode == 0:
                    args_dict = vars(args)
                    for k, v in sorted(args_dict.items()):
                        print(f'{k}: {v}, ', end=' ')
                    print('')
                    print(f'>>>>>>>start training : {setting} (batch_size:{args.batch_size}, lamda1:{lamda1:.2f}, '
                          f'alpha:{args.alpha:.2f}, lossfun_alpha:{args.lossfun_alpha:.2f})'
                          f'(best_mse:{best_mse:.5f}, best_mae:{best_mae:.5f}) (Monte Carlo: {idx}/{itr_count}) '
                          f'(epochs per exp:{args.train_epochs})'
                          f'>>>>>>>>>>>>')
                    exp.train(setting)
                else:
                    if ii > 0:
                        break
                    else:
                        args_dict = vars(args)
                        for k, v in sorted(args_dict.items()):
                            print(f'{k}: {v}, ', end=' ')
                        print('')

                # multiple choices are
                mse = mae = math.inf
                best_batch_size = np.nan
                for test_bs in sorted(test_batch_size_list):
                    print('>>>>>>>testing : {} (test_batch_size: {})<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.
                          format(setting, test_bs))
                    mse0, mae0 = exp.test(setting, test=args.test_mode, test_batch_size=test_bs)

                    if mse0 < mse:
                        mse, mae = mse0, mae0
                        best_batch_size = test_bs
                print(f'\tbest_test_batch_size: {best_batch_size}, best_mse: {mse:.5f}, best_mae: {mae:.5f}')

                if mse + mae <= best_mse + best_mae:
                    best_lamda1 = lamda1
                    best_mse, best_mae, best_ii = mse, mae, ii

                # log into txt
                mse_mse_string = (f'mse:{mse:.5f}, mae:{mae:.5f}, lamda1:{lamda1:.2f}, '
                                  f'git_multi_stage:{args.git_multi_stage}, decoder_cat:{args.decoder_cat_num}, '
                                  f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}')
                print(mse_mse_string)
                with open(log_txt, 'a') as f:
                    f.write(f'------------ {setting} -------------' + '\n' + '\n')
                    args_dict = vars(args)
                    for k, v in sorted(args_dict.items()):
                        f.write(f'{k}: {v}, ')
                    f.write('\n\n')
                    f.write('\t' + mse_mse_string + '\n\n')
                    f.write('--------------------------------- Ends -----------------------------\n\n')

                time_vec.append(time.time() - time_now)
                print(f'training time left is {np.mean(time_vec) * (itr_count - idx) / 60.0:.2f} min...')
                # torch.cuda.empty_cache()

                # best_string by far
                best_string = (f'best_mse (by far): {best_mse:.5f}, best_mae: {best_mae:.5f};\t '
                               f'best_lamda1: {best_lamda1:.2f}, \t'
                               f'git_multi_stage:{args.git_multi_stage}, '
                               f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}, '
                               f'best_ii:{best_ii}, used time: {time_vec[-1] / 60.0: .2f} min(s)')
                print(best_string)

            # write into txt
            with open(log_txt, 'a') as f:
                f.write(f'============================= {args.model_id}============================= \n')
                f.write('\n\t' + best_string + '\n\n\n')
                f.write('================================== end ===================================\n\n')

            with open(best_log_txt, 'a') as f:
                f.write(f'============================= {args.model_id}=============================\n')
                f.write('\n\t' + best_string + '\n' + '\n')
                f.write('================================== end ===================================\n\n')

            args.lamda1 = lamda1_ori

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

        # global best
        best_string = (f'global best_mse: {best_mse:.5f}, best_mae: {best_mae:.5f};\t '
                       f'best_lamda1: {best_lamda1:.2f},  \t'
                       f'git_multi_stage:{args.git_multi_stage}, '
                       f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}, '
                       f'best_ii:{best_ii}; Avg time: {np.mean(time_vec) / 60.0: .2f} min(s)')
        print(best_string)
        with open(best_log_txt, 'a') as f:
            f.write(f'============================ global best of {model_id_ori}=============================\n')
            f.write('\n\t' + best_string + '\n\n')
            f.write('======================================== end ============================================\n\n')
        # torch.cuda.empty_cache()

        # write also into best_log_dataset_txt
        with open(best_log_dataset_txt, 'a') as f:
            f.write(f'============================ global best of {args.model_id}=============================\n')
            args_dict = vars(args)
            for k, v in sorted(args_dict.items()):
                f.write(f'\t{k}: {v}; ')
            f.write('\n')
            f.write('\n\t' + best_string + '\n\n')
            f.write('======================================== end ============================================'
                    + '\n' + '\n')

    elif not args.model_stats_mode and not args.is_training and not args.save_linear_weight:
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            print(f'{k}: {v}, ', end=' ')
        print('')

        ii = 0
        setting = get_setting(args, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    elif args.model_stats_mode and not args.save_linear_weight:
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            print(f'{k}: {v}, ', end=' ')
        print('')
        print(f'============================ model_stats {args.model_id_ori}============================= ')
        exp = Exp(args)  # set experiments
        exp.compute_model_stats()

    elif args.save_linear_weight:
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            print(f'{k}: {v}, ', end=' ')
        print('')
        print(f'============================ save_linear_weight {args.model_id_ori}============================= ')
        exp = Exp(args)  # set experiments
        exp.save_linear_weight_2npy(setting=setting_zero)

    print(f'A total of {(time.time() - global_time0) / 60.0: .2f} min(s) used...')

    if args.send_mail and not args.model_stats_mode and args.is_training and args.test_mode == 0:
        # only send mail after training
        mess_body = (f'{args.task_name}_{args.model_id} program complete. MSE: {best_mse:.5f}, MAE: {best_mae:.5f}. '
                     f'Avg time: {np.mean(time_vec) / 60.0: .2f} min(s).')
        send_email(body=mess_body)
