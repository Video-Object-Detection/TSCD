#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pywt
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch

from yolox.utils.feature_visualization import feature_visualization


class DWT_Function(Function):
    """
        Discrete Wavelet Transform (DWT) function
    """
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    """
    Inverse Discrete Wavelet Transform (IDWT) function
    """
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    """
    2D Inverse Discrete Wavelet Transform
    """
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters.type_as(x))


class DWT_2D(nn.Module):
    """
    2D Discrete Wavelet Transform
    """
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll.type_as(x), self.w_lh.type_as(x), self.w_hl.type_as(x),
                                  self.w_hh.type_as(x))


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class WaveletsBlock(nn.Module):
    def __init__(self, in_channels):
        super(WaveletsBlock, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      stride=2, groups=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        o_x = x.reshape(B, H * W, C)
        x_dwt = self.dwt(x)
        # edge = self.filter(x_dwt)
        new_x = self.filter(x)
        edge = new_x.repeat(1, 4, 1, 1) * x_dwt
        x_idwt = self.idwt(edge).reshape(B, H * W, C)
        x = self.proj(torch.cat([o_x, x_idwt], dim=-1))
        x = x.reshape(B, C, H, W)
        return x


class WaveletsLFBlock(nn.Module):
    """
        use the DWT extract the core low frequency features
    """
    def __init__(self, in_channels):
        super(WaveletsLFBlock, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.filter1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        C = x.shape[1]
        x_dwt = self.dwt(x)
        LF, HF = x_dwt.split([C, C*3], dim=1)
        HF = torch.zeros_like(HF)
        LF = self.filter1(LF)
        x_dwt = torch.cat((LF, HF), dim=1)
        x_idwt = self.idwt(x_dwt)
        x_content = self.filter2(x)
        x_edge_content = x_content * x_idwt
        return x_edge_content


class WaveletsHFBlock(nn.Module):
    """
        use the DWT extract the surrounding high frequency features
    """
    def __init__(self, in_channels):
        super(WaveletsHFBlock, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.filter1 = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels*3, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )

    def forward_visual(self, x):
        """
            visual the feature map
        """
        feature_visualization(x, 'WaveletsHFBlock_start', 1)
        C = x.shape[1]
        x_dwt = self.dwt(x)
        LF, HF = x_dwt.split([C, C*3], dim=1)
        feature_visualization(LF, 'WaveletsHFBlock_dwt_lf', 1)
        feature_visualization(HF, 'WaveletsHFBlock_dwt_hf', 1)
        # LF = torch.zeros_like(LF)
        HF = self.filter1(HF)
        feature_visualization(HF, 'WaveletsHFBlock_dwt_hf_filer1', 1)
        x_dwt = torch.cat((LF, HF), dim=1)
        feature_visualization(x_dwt, 'WaveletsHFBlock_dwt_lf_zero', 1)
        x_idwt = self.idwt(x_dwt)
        feature_visualization(x_idwt, 'WaveletsHFBlock_idwt', 1)
        x_content = self.filter2(x)
        feature_visualization(x_content, 'WaveletsHFBlock_x_filter2', 1)
        x_edge_content = x_content * x_idwt
        feature_visualization(x_edge_content, 'WaveletsHFBlock_x_edge_content', 1)
        return x_edge_content

    def forward(self, x):
        C = x.shape[1]
        x_dwt = self.dwt(x)
        LF, HF = x_dwt.split([C, C*3], dim=1)
        LF = torch.zeros_like(LF)
        HF = self.filter1(HF)
        x_dwt = torch.cat((LF, HF), dim=1)
        x_idwt = self.idwt(x_dwt)
        x_content = self.filter2(x)
        x_edge_content = x_content * x_idwt
        return x_edge_content


class WaveletsClsBlock(nn.Module):
    def __init__(self, in_channels):
        super(WaveletsClsBlock, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(in_channels * 2, in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        o_x = x.reshape(B, H * W, C)
        x_dwt = self.dwt(x)
        LF, HF = x_dwt.split([128, 384], dim=1)
        LF = torch.zeros_like(LF)
        # HF = self.filter(HF)
        x_dwt = torch.cat((LF, HF), dim=1)
        x_idwt = self.idwt(x_dwt)
        x_content = self.filter(x)
        x_edge_content = x_content * x_idwt
        x_edge_content = x_edge_content.reshape(B, H * W, C)
        x = self.proj(torch.cat([o_x, x_edge_content], dim=-1))
        x = x.reshape(B, C, H, W)
        # x = self.act(x)
        return x


class WaveletsRegBlock(nn.Module):
    def __init__(self, in_channels):
        super(WaveletsRegBlock, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.filter1 = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.filter3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0,
                      stride=1, groups=1),
            nn.ReLU(inplace=True),
        )

        # self.proj = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_dwt = self.dwt(x)
        LF, HF = x_dwt.split([128, 384], dim=1)
        LF = self.filter3(LF)
        HF = self.filter1(HF) + self.filter2(HF)
        # HF_Z = torch.zeros_like(HF)
        x_dwt = torch.cat([LF, HF], dim=1)
        x_enhance = self.idwt(x_dwt)
        # x = self.proj(torch.cat([x.reshape(B, H * W, C), x_enhance.reshape(B, H * W, C)], dim=-1))
        # x = x.reshape(B, C, H, W)
        x = 0.5 * x + 0.5 * x_enhance
        return x
