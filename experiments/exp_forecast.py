from torch.optim import lr_scheduler
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import (EarlyStopping, adjust_learning_rate, visual, write_into_xls, compute_gradient_norm,
                         find_most_recently_modified_subfolder, compare_prefix_before_third_underscore, compute_weights)
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim
import os
import time
import warnings
import numpy as np
from typing import List
import random

# from thop import profile
# from thop import clever_format
#
# from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
import logging
import shutil

# from torchprofile import profile_macs

warnings.filterwarnings('ignore')


def compute_model_stats(model, args, num_iterations=50):
    assert num_iterations > 10, 'num_iterations should be greater than 10'
    if not args.model_stats_mode:
        print('No compute_model_stats because model_stats_mode is False!')
        return False

    logging.getLogger('fvcore').setLevel(logging.ERROR)
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot measure GPU memory and timings.")
        return False

    device = torch.device("cuda")

    input_size = (1, args.seq_len, args.enc_in)
    inputs = torch.randn(input_size).to(device)

    # gpu
    model = model.to(device).eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters(M): {params:.3f}")

    # use fvcore to get flops
    flops = FlopCountAnalysis(model, inputs)
    flops = flops.total() / 1e6

    print(f"FLOPS(M): {flops:.3f}")

    if 'PEMS' in args.data:
        num_iterations = 15

    #  training and inferring
    inputs = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)

    # inference, make gpu memory more precise
    torch.cuda.reset_peak_memory_stats()
    inference_times = []
    for i in range(num_iterations):
        start_time = time.time()
        if args.use_amp:
            with torch.cuda.amp.autocast():
                _ = model(inputs)
        else:
            with torch.no_grad():
                _ = model(inputs)

        inference_times.append(time.time() - start_time)
    avg_inference_time = np.mean(inference_times[-10:])
    inference_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Inference Time / iter: {avg_inference_time * 1000:.3f} ms")
    print(f"Inference Memory Usage: {inference_memory:.3f} MB")

    # training
    criterion = WeightedL1Loss(args.lossfun_alpha, args.loss_mode)
    targets = torch.randn(args.batch_size, args.pred_len, args.enc_in).to(device)
    # print(inputs.shape, targets.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    training_times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_iterations):
        model.train()
        # torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        scaler = None
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        training_times.append(time.time() - start_time)
    avg_training_time = np.mean(training_times[-10:])
    training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 保存结果到文件
    with open('model_stats.txt', 'a') as f:
        f.write(f'============================ stats {args.model_id_ori}============================= ' + '\n')
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            f.write(f'{k}: {v}, ')
        f.write('\n\n')
        f.write(f"\tParameters(M): {params:.3f}\n")
        f.write(f"\tFLOPS(M): {flops:.3f}\n")
        f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
        f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
        f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
        f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

    # save to folder best_results
    best_log_dataset_path = 'best_results'
    best_log_dataset_txt = os.path.join(best_log_dataset_path, args.model_id_ori + '_stats.txt')
    with open(best_log_dataset_txt, 'a') as f:
        f.write(f'============================ stats {args.model_id_ori}============================= ' + '\n')
        args_dict = vars(args)
        for k, v in sorted(args_dict.items()):
            f.write(f'{k}: {v}, ')
        f.write('\n\n')
        f.write(f"\tParameters(M): {params:.3f}\n")
        f.write(f"\tFLOPS(M): {flops:.3f}\n")
        f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
        f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
        f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
        f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

    # 打印结果

    print(f"Training Time / iter: {avg_training_time * 1000:.3f} ms")
    print(f"Training Memory Usage: {training_memory:.3f} MB")

    return True


class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # [b,l,n]
        if pred.ndim == 1:
            # imputation
            mask = torch.isnan(gt)
            if torch.any(mask):
                # pred, gt = pred.masked_fill(mask, 0), gt.masked_fill(mask, 0)
                pred, gt = pred[~mask], gt[~mask]

            loss_fun = nn.L1Loss(reduction='mean')
            weightedLoss = loss_fun(pred, gt)
        else:
            L = pred.shape[1]
            weights = (torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)]).unsqueeze(dim=0).unsqueeze(dim=-1)
                       .to(pred.device))
            if self.loss_mode in ['L1', 'L2']:
                loss_vec = self.loss_fun(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights)
            elif self.loss_mode == 'L1L2':
                loss_vec = self.loss_fun1(pred, gt)
                loss_vec2 = self.loss_fun2(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
            else:
                raise NotImplementedError
        return weightedLoss


def _select_criterion():
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss(reduction='mean')
    return criterion


def _select_mse_criterion():
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss(reduction='mean')
    return criterion


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.imp_mode = args.task_name == 'imputation'
        self.eval_flag = args.eval_flag
        self.resume_training = args.resume_training
        self.resume_epoch = args.resume_epoch
        self.folder_path = args.folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path, exist_ok=True)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def compute_model_stats(self):
        # model_stats
        if self.args.model_stats_mode:
            compute_model_stats(self.model, self.args)

    def save_linear_weight_2npy(self, setting=None):
        assert self.args.save_linear_weight
        # model_stats
        full_folder, new_setting = find_most_recently_modified_subfolder(self.args.checkpoints,
                                                                         file_name='checkpoint.pth',
                                                                         contain_str=self.args.model_id_ori)
        if full_folder is not None and compare_prefix_before_third_underscore(setting, new_setting):
            print(f'loading model from {full_folder}')
            self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
        else:
            raise ValueError('No checkpoints are found...')

        self.model.eval()

        # nothing interesting found in first_linear_layer.weight
        # ortho_trans_module = self.model.ortho_trans
        # # obtain first_linear_layer
        # first_linear_layer = ortho_trans_module[0]

        weight_mat = self.model.encoder.attn_layers[0].weight_mat[0, ...]

        # softplus L1
        weight_mat = F.normalize(F.softplus(weight_mat), p=1, dim=-1)

        weight_numpy = weight_mat.detach().cpu().numpy()

        # make dirs if it does not exist
        if not os.path.exists(self.args.save_linear_weight_path):
            os.makedirs(self.args.save_linear_weight_path)

        output_filename = os.path.join(self.args.save_linear_weight_path,
                                       f'{self.args.save_linear_weight_tag}_normlin_weight_after_norm.npy')

        np.save(output_filename, weight_numpy)

        print(f"The weight is saved to '{output_filename}'")
        print(f"Weight shape: {weight_numpy.shape}")

        loaded_weights = np.load(output_filename)
        print("Weight (first 5 entries):", loaded_weights[0, :5])

    def _get_data(self, flag=None, test_batch_size=None):
        data_set, data_loader = data_provider(self.args, flag, test_batch_size=test_batch_size)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None, criterion=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                B, T, N = batch_x.shape

                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                mask_input = None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=mask_input)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=mask_input)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid', test_batch_size=self.args.batch_size)
        test_data, test_loader = self._get_data(flag='test', test_batch_size=self.args.batch_size)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       save_every_epoch=self.args.save_every_epoch)

        model_optim = self._select_optimizer()
        criterion = WeightedL1Loss(self.args.lossfun_alpha, self.args.loss_mode)
        criterion_no_decay = None
        if self.args.Q_loss:
            criterion_no_decay = WeightedL1Loss(0, 'L1')

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.resume_training and self.resume_epoch > 0:
            full_folder, new_setting = find_most_recently_modified_subfolder(self.args.checkpoints,
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=[self.args.model_id_ori,
                                                                                          self.args.model])
            if compare_prefix_before_third_underscore(setting, new_setting, num=3):
                print(f'loading model from {full_folder}')
                self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
                shutil.copy(os.path.join(full_folder, 'checkpoint.pth'), path)
            else:
                raise ValueError('No checkpoint folder found. Please check...')

            current_val_loss = self.vali(vali_data, vali_loader, criterion)
            early_stopping.best_score = -current_val_loss
            early_stopping.val_loss_min = current_val_loss

        start_epoch = self.resume_epoch if self.resume_training else 0
        if self.resume_training:
            print('Restoring the learning rate...')
            adjust_learning_rate(model_optim, start_epoch, self.args)

        Q_out_mat = None
        if self.args.Q_loss:
            q_out_mat_dir = os.path.join(self.args.root_path, self.args.q_out_mat_file)
            assert os.path.isfile(q_out_mat_dir)
            Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(self.device)

        scheduler = None
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                B, T, N = batch_x.shape

                batch_y = batch_y.float().to(self.device)

                if self.args.efficient_training:  # and self.args.e_layers > 0 and self.args.second_e_layers > 0:
                    _, _, N = batch_x.shape
                    if N > self.args.enc_in:
                        index = np.stack(random.sample(range(N), self.args.enc_in))
                        batch_x = batch_x[:, :, index]
                        batch_y = batch_y[:, :, index]

                # encoder - decoder
                mask_input = None
                if self.args.use_amp:
                    with (torch.cuda.amp.autocast()):
                        outputs0 = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=mask_input)

                        if isinstance(outputs0, tuple):
                            outputs = outputs0[0]
                        else:
                            outputs = outputs0

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.Q_loss and self.args.Q_loss_alpha >= 1:
                            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        else:
                            loss = criterion(outputs, batch_y)

                        if self.args.Q_loss:
                            outputs_trans = torch.einsum('btn,tv->bvn', outputs, Q_out_mat.transpose(-1, -2))
                            batch_y_trans = torch.einsum('btn,tv->bvn', batch_y, Q_out_mat.transpose(-1, -2))
                            loss_q = criterion_no_decay(outputs_trans, batch_y_trans)
                            loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_q

                        if isinstance(outputs0, tuple) and len(outputs0) >= 3:
                            dec_out_inter = outputs0[2]
                            if dec_out_inter:
                                if isinstance(dec_out_inter, List):

                                    loss1_vec = torch.stack([criterion(seq, batch_y)
                                                             for seq in dec_out_inter])
                                    weights = compute_weights(self.args.alpha, len(loss1_vec),
                                                              self.args.git_multi_stage + 1, multiple_flag=self.imp_mode
                                                              ).to(self.device)
                                    loss1 = (loss1_vec * weights).sum()

                                else:
                                    loss1 = criterion(dec_out_inter, batch_y)

                                loss = loss + self.args.lamda1 * loss1

                        train_loss.append(loss.item())
                else:
                    outputs0 = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=mask_input)
                    if isinstance(outputs0, tuple):
                        outputs = outputs0[0]
                    else:
                        outputs = outputs0

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape, self.args.pred_len, f_dim)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # back to pred_len
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.Q_loss and self.args.Q_loss_alpha >= 1:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    else:
                        loss = criterion(outputs, batch_y)

                    if self.args.Q_loss:
                        outputs_trans = torch.einsum('btn,tv->bvn', outputs, Q_out_mat.transpose(-1, -2))
                        batch_y_trans = torch.einsum('btn,tv->bvn', batch_y, Q_out_mat.transpose(-1, -2))
                        loss_q = criterion_no_decay(outputs_trans, batch_y_trans)
                        loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_q
                    elif self.args.FFT_loss:
                        outputs_fft = torch.fft.rfft(outputs, dim=1)
                        batch_y_fft = torch.fft.rfft(batch_y, dim=1)
                        loss_fft = torch.mean(torch.abs(outputs_fft - batch_y_fft))
                        loss = (1 - self.args.Q_loss_alpha) * loss + self.args.Q_loss_alpha * loss_fft

                    if isinstance(outputs0, tuple) and len(outputs0) >= 2:
                        dec_out_inter = outputs0[2]
                        if dec_out_inter:
                            if isinstance(dec_out_inter, List):
                                loss1_vec = torch.stack([criterion(seq, batch_y)
                                                         for seq in dec_out_inter])
                                weights = (compute_weights(self.args.alpha, len(loss1_vec),
                                                           self.args.git_multi_stage + 1,
                                                           multiple_flag=self.imp_mode)
                                           .to(self.device))
                                loss1 = (loss1_vec * weights).sum()
                            else:
                                loss1 = criterion(dec_out_inter, batch_y)

                            loss = loss + self.args.lamda1 * loss1

                    if torch.isnan(loss):
                        print('\tloss is nan. please check...')
                        print("\toutputs.shape: ", outputs.shape, '\tbatch_y.shape: ', batch_y.shape)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0 or i == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:d} min, {:.2f} s'.format(speed,
                                                                                        int(left_time // 60),
                                                                                        left_time % 60))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    model_optim.zero_grad()
                    scaler.scale(loss).backward()

                    # grad emerges after backward
                    if (i + 1) % 100 == 0 or i == 0:
                        grd_norm = compute_gradient_norm(self.model)
                        if self.args.grad_clip:
                            print(f"\t\tTotal norm of gradients: {grd_norm:.2f}, "
                                  f"{'clipped!' if grd_norm > self.args.max_norm else ''}")
                        else:
                            print(f"\t\tTotal norm of gradients: {grd_norm:.2f}")
                    if self.args.grad_clip:
                        clip_grad_norm_(self.model.parameters(), self.args.max_norm)

                    scaler.step(model_optim)
                    scaler.update()
                else:

                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, False)
                    scheduler.step()

            t2 = time.time() - epoch_time
            print("Epoch: {} cost time: {}min {:.1f}s".format(epoch + 1, t2 // 60, t2 % 60))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path, epoch=epoch + 1)
            if early_stopping.early_stop:  #
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting=None, test=0, test_batch_size=None):
        test_data, test_loader = self._get_data(flag='test', test_batch_size=test_batch_size)
        if test < 1:
            print('loading model...')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        else:
            full_folder, new_setting = find_most_recently_modified_subfolder(self.args.checkpoints,
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=self.args.model_id_ori)
            if full_folder is not None and compare_prefix_before_third_underscore(setting, new_setting):
                print(f'loading model from {full_folder}')
                self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
            else:
                raise ValueError('check most_recently_modified_subfolder')

        preds = []
        trues = []
        inters = []
        eval_imp = []
        dec_seq_inter = []
        mask_mat_list = []
        folder_path = self.args.folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_period = max(len(test_loader) // 5, 1)
        eval_stages = self.args.git_multi_stage + 1
        self.model.eval()
        if hasattr(self.model, 'alpha'):
            print(f'self.model.alpha: {self.model.alpha}')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    if not self.imp_mode:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                B, T, N = batch_x.shape
                if self.imp_mode:
                    if self.args.data.lower() == 'air':

                        # not random
                        # obs_mat = batch_y.to(self.device)
                        # batch_y = batch_x
                        #
                        # gt_mask = batch_x_mark.to(self.device)
                        # batch_y[~obs_mat] = torch.nan
                        #
                        # mask_mat = ~gt_mask

                        #  # random settings
                        obs_mat = batch_y.to(self.device)
                        batch_y = batch_x

                        batch_y[~obs_mat] = torch.nan

                        # batch_x[~obs_mat] = 0

                        # print(f'batch_x: {batch_x[0, 0, :]}')
                        # print(f'Is there nan in batch_x? {torch.any(torch.isnan(batch_x))}')

                        # add more missing data
                        mask = torch.rand_like(batch_x) * obs_mat  # for un-observed, mask has to be 0
                        mask[mask <= self.args.mask_rate] = 0  # masked
                        mask[mask > self.args.mask_rate] = 1  # remained
                        mask_mat = mask == 0  # 1 if masked
                        assert torch.all(mask_mat.float() + obs_mat.float())  # mask_mat must cover the nan areas

                        # gt_mask
                        # mask_mat = ~batch_x_mark.to(self.device)
                        # batch_x[mask_mat] = 0

                        batch_x = batch_x.masked_fill(mask_mat, 0)

                    elif self.args.data.lower() == 'physio':
                        obs_mat = (batch_y > 0).to(self.device)

                        batch_y = batch_x.to(self.device)

                        batch_y[~obs_mat] = torch.nan

                        # add more missing data
                        mask = torch.rand_like(batch_x) * obs_mat
                        mask[mask <= self.args.mask_rate] = 0  # masked
                        mask[mask > self.args.mask_rate] = 1  # remained
                        mask_mat = mask == 0  # 1 if masked

                        batch_x = batch_x.masked_fill(mask_mat, 0)
                    else:
                        # random mask; unique to imputation
                        batch_y = batch_x
                        batch_y_mark = batch_x_mark
                        mask = torch.rand((B, T, N)).to(self.device)
                        mask[mask <= self.args.mask_rate] = 0  # masked
                        mask[mask > self.args.mask_rate] = 1  # remained
                        mask_mat = mask == 0  # 1 for missing values
                        batch_x = batch_x.masked_fill(mask_mat, 0)
                else:
                    batch_y = batch_y.float().to(self.device)
                    # all the tensor
                    mask_mat = torch.ones((B, T, N)).to(self.device) != 0

                # encoder - decoder
                mask_input = mask_mat if self.imp_mode else None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask_input)
                else:
                    outputs_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask_input)

                if isinstance(outputs_list, tuple):
                    outputs = outputs_list[0]
                else:
                    outputs = outputs_list

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()

                if isinstance(outputs_list, tuple):
                    dec_seq_inter = outputs_list[-1]
                    dec_seq_inter = [seq[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                                     for seq in dec_seq_inter]
                    if self.imp_mode and self.eval_flag:
                        # [b,stages]
                        eval_imp_batch = outputs_list[1].detach().cpu().numpy()
                        eval_stages = eval_imp_batch.shape[-1]

                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    dec_seq_inter = [test_data.inverse_transform(seq.squeeze(0)).reshape(shape)
                                     for seq in dec_seq_inter]

                pred = outputs
                true = batch_y

                mask_mat = mask_mat.cpu().numpy()

                if self.imp_mode:
                    pred[~mask_mat] = true[~mask_mat]

                preds.append(pred)
                trues.append(true)
                inters.append(dec_seq_inter)
                if self.imp_mode and self.eval_flag:
                    eval_imp.append(eval_imp_batch)
                mask_mat_list.append(mask_mat)
                if i % save_period == 0:
                    print(f'processing batch{i} at test phase...')
                    input_ = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_.shape
                        input_ = test_data.inverse_transform(input_.squeeze(0)).reshape(shape)

                    if self.imp_mode:
                        if self.args.data == 'physio':
                            nan_mask = np.isnan(true)
                            nan_counts = np.sum(nan_mask, axis=1)
                            min_ind = np.argmin(nan_counts)

                            it, jt = min_ind // true.shape[-1], min_ind % true.shape[-1]
                            gt = true[it, :, jt]
                            pd = pred[it, :, jt]
                        else:
                            gt = true[0, :, -1]
                            # pd = gt * (~mask_mat[0, :, -1]) + pred[0, :, -1] * mask_mat[0, :, -1]
                            pd = pred[0, :, -1]  # only pred
                    else:
                        gt = np.concatenate((input_[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_[0, :, -1], pred[0, :, -1]), axis=0)

                    if self.args.save_pdf:
                        # save pdf every 20 batches [the first time series in the batch and the last dimension]
                        save_folder = os.path.join(folder_path, 'pred-gt-pdf-npy')
                        visual(gt, pd, os.path.join(save_folder,
                                                    f'{self.args.model_id}_batch_{i}_imp-gt.pdf'
                                                    if self.imp_mode else f'{self.args.model_id}_batch_{i}_pred-gt.pdf'),
                               imp=self.imp_mode)
                        np.save(os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_pred.npy'), pd)
                        np.save(os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_gt.npy'), gt)
                        write_into_xls(excel_name=os.path.join(save_folder,
                                                               f'{self.args.model_id}_batch_{i}_gt.xlsx'),
                                       mat=gt, columns=None)
                        write_into_xls(excel_name=os.path.join(save_folder,
                                                               f'{self.args.model_id}_batch_{i}_pred.xlsx'),
                                       mat=pd, columns=None)

                        for ii, seq_inter in enumerate(dec_seq_inter):

                            if self.imp_mode:
                                seq_inter[~mask_mat] = true[~mask_mat]
                                if self.args.data == 'physio':
                                    pdi = seq_inter[it, :, jt]
                                else:
                                    pdi = seq_inter[0, :, -1]
                            else:
                                pdi = np.concatenate((input_[0, :, -1], seq_inter[0, :, -1]), axis=0)
                            visual(gt, pdi, os.path.join(save_folder,
                                                         f'{self.args.model_id}_batch_{i}_imp-gt_stage_{ii}.pdf'
                                                         if self.imp_mode else
                                                         f'{self.args.model_id}_batch_{i}_pred-gt_stage_{ii}.pdf'),
                                   imp=self.imp_mode)
                            np.save(os.path.join(save_folder, f'{self.args.model_id}_batch_{i}_pred_stage_{ii}.npy'),
                                    pdi)
                            write_into_xls(excel_name=os.path.join(save_folder,
                                                                   f'{self.args.model_id}_batch_{i}_pred_stage_{ii}.xlsx'),
                                           mat=pdi, columns=None)

        # preds_array = np.array(preds)
        preds_array = np.concatenate(preds, axis=0)
        trues_array = np.concatenate(trues, axis=0)
        mask_mat = np.concatenate(mask_mat_list, axis=0)
        if self.imp_mode and self.eval_flag:
            eval_imp = np.array(eval_imp).reshape((-1, eval_stages)).mean(axis=0)  # [stages]
        print('test shape:', preds_array.shape, trues_array.shape)

        mse_list = []
        mae_list = []

        if self.imp_mode:
            mae, mse, rmse, mape, mspe = metric(preds_array[mask_mat], trues_array[mask_mat])
            f = open("result_imputation.txt", 'a')
        else:
            mae, mse, rmse, mape, mspe, r2, pear, mase = metric(preds_array, trues_array)
            print(f'preds_array.shape: {preds_array.shape}')
            print(f'final output mse: {mse:.5f}, mae: {mae:.5f}, r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}')
            f = open("result_long_term_forecast.txt", 'a')

        # print seq_inter
        if isinstance(outputs_list, tuple) and self.args.seq_inter:
            if self.imp_mode:
                true_list = []
                pred_list = []
                for inter_seq_list, true, mask_mat in zip(inters, trues, mask_mat_list):
                    true_list.append(true[mask_mat])
                    pred_list.append([inter_seq[mask_mat] for inter_seq in inter_seq_list])
                # transpose
                all_pred_list = list(zip(*pred_list))

                true_arr = np.concatenate([arr.flatten() for arr in true_list])
                all_pred_arr = [np.concatenate([arr.flatten() for arr in pred_stage]) for pred_stage in all_pred_list]
                # print(len(all_pred_arr))

                all_stages = len(all_pred_arr) + 1
                stage_num = all_stages // (self.args.git_multi_stage + 1)
                for i, pred in enumerate(all_pred_arr):
                    if i == 0:
                        print('stage (i) seq shape:', pred.shape)
                    elif i % stage_num == 0:
                        print('----------------------------------')
                    mae_, mse_, _, _, _, _, _, _ = metric(pred, true_arr)
                    mse_list.append(mse_)
                    mae_list.append(mae_)
                    if self.eval_flag:
                        print(f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}, '
                              f'cos_dist:{eval_imp[min(i, len(eval_imp) - 1)]:.5f}')
                    else:
                        print(f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}')

            else:
                stage_list = list(zip(*inters))
                stage_list = [np.concatenate(_, axis=0) for _ in stage_list]
                for i, pred in enumerate(stage_list):
                    if i == 0:
                        print('stage (i) seq shape:', pred.shape)
                    mae_, mse_, _, _, _, r2_, pear_, mase_ = metric(pred, trues_array)
                    mse_list.append(mse_)
                    mae_list.append(mae_)
                    print(f'stage{i}: mse:{mse_:.5f}, mae:{mae_:.5f}, '
                          f'r2: {r2_:.5f}, pear: {pear_:.5f}, mase: {mase_:.5f}')

        if self.imp_mode and self.eval_flag:
            print(f'stage{len(mse_list)}: mse:{mse:.5f}, mae:{mae:.5f}, cos_dist:{eval_imp[-1]:.5f}')
        else:
            if self.imp_mode:
                print(f'stage{len(mse_list)}: mse:{mse:.5f}, mae:{mae:.5f}')
            else:
                print(f'stage{len(mse_list)}: mse:{mse:.5f}, mae:{mae:.5f}, '
                      f'r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}')
        mse_list.append(mse)
        mae_list.append(mae)

        mse_plus_mae = [a + b for a, b in zip(mse_list, mae_list)]

        # print min_mse
        # print(mse_list, mae_list)
        min_index = mse_plus_mae.index(min(mse_plus_mae))
        mse = mse_list[min_index]
        mae = mae_list[min_index]

        f.write(setting + "  \n")
        if self.imp_mode:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        else:
            f.write(f'mae: {mae:.5f}, mse: {mse:.5f}, r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}')
        f.write('\n')
        f.write('\n')
        f.close()

        if self.imp_mode and self.eval_flag:
            cos_dist = eval_imp[min(min_index, len(eval_imp) - 1)]  # cos_dist could be short
            print(f"Of all stages, best stage: {min_index}; best mse:{mse:.5f}, best mae:{mae:.5f}, "
                  f"cos_dist:{cos_dist:.5f}")
        else:
            print(f"Of all stages, best stage: {min_index}; best mse:{mse:.5f}, best mae:{mae:.5f}")

        # to npy files
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        # np.save(os.path.join(folder_path, 'pred.npy'), preds)
        # np.save(os.path.join(folder_path, 'true.npy'), trues)

        # to excel
        write_into_xls(os.path.join(folder_path, 'metrics.xlsx'), [mae, mse, rmse, mape, mspe])
        # columns=['mae', 'mse', 'rmse', 'mape', 'mspe']
        # write_into_xls(folder_path + 'pred.xlsx', preds)
        # write_into_xls(folder_path + 'true.xlsx', trues)

        # rename
        file_name = f"MSE_{mse:.5f}_MAE_{mae:.5f}_" + setting
        new_folder_path = os.path.join('results', file_name[:254])
        os.rename(folder_path, new_folder_path)

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape  # [b,s,n]
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join('results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return
