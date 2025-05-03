import os
import torch
from model import OLinear, OLinear_C, OLinear_attn_var, OLinear_ablation_var_temp, OLinear_ablation_lin_design, \
    OLinear_no_Q_neither

from model.orthoLinear_basis import OLinear_FFT, OLinear_wavelet, OLinear_wavelet2, OLinear_Legendre, OLinear_Laguerre, \
    OLinear_cheby


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'OLinear': OLinear,
            'OLinear_C': OLinear_C,
            'OLinear_attn_var': OLinear_attn_var,
            'OLinear_ablation_var_temp': OLinear_ablation_var_temp,
            'OLinear_ablation_lin_design': OLinear_ablation_lin_design,
            'OLinear_no_Q_neither': OLinear_no_Q_neither,
            'OLinear_FFT': OLinear_FFT,
            'OLinear_wavelet': OLinear_wavelet,
            'OLinear_wavelet2': OLinear_wavelet2,
            'OLinear_Legendre': OLinear_Legendre,
            'OLinear_Laguerre': OLinear_Laguerre,
            'OLinear_cheby': OLinear_cheby,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
