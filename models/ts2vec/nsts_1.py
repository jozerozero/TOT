import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
from einops import rearrange, reduce, repeat

from .fsnet_ import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_norm - 2):
                net.append(nn.Linear(in_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Base_Net(nn.Module):
    def __init__(self, input_len, out_len, input_dim, out_dim, is_mean_std=True, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len
        self.c_type = c_type
        self.out_dim = out_dim
        self.c_type = "type1" if out_dim != input_dim and c_type == "None" else c_type
        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=out_len * 2,
                           activation=activation,
                           layer_norm=layer_norm)
        elif self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=self.out_dim * 2,
                           out_dim=self.out_dim * self.radio,
                           layer_norm=layer_norm, activation=activation)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=self.out_dim * 2 * input_len,
                           activation=activation,
                           out_dim=self.out_dim * input_len * self.radio, layer_norm=layer_norm)

        self.dropout_net = nn.Dropout(drop_out)

    def forward(self, x):
        if self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim)

        elif self.c_type == "None":
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout_net(x)
        if self.radio == 2:
            dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
            x = torch.chunk(x, dim=dim, chunks=2)
        return x


class Encoder_ZS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_zs_mean = Base_Net(self.args.seq_len, self.seq_len, self.feature_dim,
                                self.z_dim * 2,
                                layer_norm=self.is_ln, activation=self.activation,
                                drop_out=self.dropout, layer_nums=self.layer_nums)



class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10

    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    # def forward_zs_mean(self, x, mask=None):  # x: B x T x input_dims
    #     x = x.transpose(1, 2)
    #     nan_mask = ~x.isnan().any(axis=-1)
    #     x[~nan_mask] = 0
    #     x = self.input_fc(x)  # B x T x Ch
    #
    #     # generate & apply mask
    #     if mask is None:
    #         if self.training:
    #             mask = self.mask_mode
    #         else:
    #             mask = 'all_true'
    #
    #     if mask == 'binomial':
    #         mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
    #     elif mask == 'continuous':
    #         mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    #     elif mask == 'all_true':
    #         mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     elif mask == 'all_false':
    #         mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    #     elif mask == 'mask_last':
    #         mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #         mask[:, -1] = False
    #
    #     mask &= nan_mask
    #     x[~mask] = 0
    #
    #     # conv encoder
    #     x = x.transpose(1, 2)  # B x Ch x T
    #     x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
    #     x = x.transpose(1, 2)  # B x T x Co
    #
    #     return x
    #
    # def forward_zs_std(self, x, mask=None):  # x: B x T x input_dims
    #     x = x.transpose(1, 2)
    #     nan_mask = ~x.isnan().any(axis=-1)
    #     x[~nan_mask] = 0
    #     x = self.input_fc(x)  # B x T x Ch
    #
    #     # generate & apply mask
    #     if mask is None:
    #         if self.training:
    #             mask = self.mask_mode
    #         else:
    #             mask = 'all_true'
    #
    #     if mask == 'binomial':
    #         mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
    #     elif mask == 'continuous':
    #         mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    #     elif mask == 'all_true':
    #         mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     elif mask == 'all_false':
    #         mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    #     elif mask == 'mask_last':
    #         mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #         mask[:, -1] = False
    #
    #     mask &= nan_mask
    #     x[~mask] = 0
    #
    #     # conv encoder
    #     x = x.transpose(1, 2)  # B x Ch x T
    #     x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
    #     x = x.transpose(1, 2)  # B x T x Co
    #
    #     return x

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


class BandedFourierLayer(nn.Module):

    def __init__(self, in_channels, out_channels, band, num_bands, freq_mixing=False, bias=True, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.freq_mixing = freq_mixing

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        if self.freq_mixing:
            self.weight = nn.Parameter(
                torch.empty((self.num_freqs, in_channels, self.total_freqs, out_channels), dtype=torch.cfloat))
        else:
            self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        else:
            self.bias = None
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        if self.freq_mixing:
            output = torch.einsum('bai,tiao->bto', input, self.weight)
        else:
            output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        if self.bias is None:
            return output
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class GlobalLocalMultiscaleTSEncoder(nn.Module):

    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 num_bands: int,
                 freq_mixing: bool,
                 length: int,
                 mode=0,
                 hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9):
        super().__init__()

        self.mode = mode

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma
        )

        self.kernels = kernels
        self.num_bands = num_bands

        self.convs = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims // 2, k, padding=k - 1) for k in kernels]
        )
        self.fouriers = nn.ModuleList(
            [BandedFourierLayer(output_dims, output_dims // 2, b, num_bands,
                                freq_mixing=freq_mixing, length=length) for b in range(num_bands)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        if len(self.kernels) == 0:
            local_multiscale = None
        else:
            local_multiscale = []
            for idx, mod in enumerate(self.convs):
                out = mod(x)  # b d t
                if self.kernels[idx] != 1:
                    out = out[..., :-(self.kernels[idx] - 1)]
                local_multiscale.append(out.transpose(1, 2))  # b t d
            local_multiscale = reduce(
                rearrange(local_multiscale, 'list b t d -> list b t d'),
                'list b t d -> b t d', 'mean'
            )

        x = x.transpose(1, 2)  # B x T x Co

        if self.num_bands == 0:
            global_multiscale = None
        else:
            global_multiscale = []
            for mod in self.fouriers:
                out = mod(x)  # b t d
                global_multiscale.append(out)

            global_multiscale = global_multiscale[0]

        return torch.cat([local_multiscale, global_multiscale], dim=-1)
