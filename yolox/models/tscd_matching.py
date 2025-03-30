#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from typing import Optional


class PositionMHAttention(nn.Module):
    """
        multi-head across-attention with the object relative position information
    """

    def __init__(self, d_model, num_heads=8, dropout=0., qkv_bias=False):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads

        self.q_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.position_embedding = nn.Conv2d(64, num_heads, kernel_size=1, stride=1, padding=0)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, q_boxes=None, k_boxes=None, attn_mask=None, key_padding_mask=None):
        x_q_reg = query
        x_k_reg = key
        x_v_reg = value
        N, B, C = x_q_reg.shape

        q_reg = self.q_reg(x_q_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        k_reg = self.k_reg(x_k_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v_reg = self.v_reg(x_v_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1))

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        if q_boxes is None or k_boxes is None:
            attn = attn_reg
        else:
            # relative position attention
            position_embedding = self.cal_position_embedding(q_boxes, k_boxes)
            # position_embedding = F.relu(self.position_embedding(position_embedding))
            position_embedding = F.relu(self.position_embedding(position_embedding.type_as(query)))
            attn = (position_embedding + 1e-6).log() + attn_reg

        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)

        return x_reg.transpose(0, 1), attn

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        """
           extract relative position embedding
        """
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        """
            extract the detection boxes relative position information
        """
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def cal_position_embedding(self, rois1, rois2):
        """
            get relative embedding
            :param rois1: x1y1x2y2
            :param rois2: x1y1x2y2
            :param feat_dim: feature dim
        """
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding


class MHAttention(nn.Module):
    """
         multi-head cross-attention
    """

    def __init__(self, d_model, num_heads=8, dropout=0., qkv_bias=False):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads

        self.q_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_reg = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        x_q_reg = query
        x_k_reg = key
        x_v_reg = value
        N, B, C = x_q_reg.shape

        q_reg = self.q_reg(x_q_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        k_reg = self.k_reg(x_k_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v_reg = self.v_reg(x_v_reg).reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1))

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = attn_reg

        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)

        return x_reg.transpose(0, 1), attn


class DoubleMHAttention(nn.Module):
    """
        double branch multi-head cross-attention
    """

    def __init__(self, d_model, num_heads=8, dropout=0., qkv_bias=False):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads

        self.q_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_reg = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.q_cls = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_cls = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_reg = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query_reg, key_reg, query_cls, key_cls, value, attn_mask=None, key_padding_mask=None):
        x_q_reg = query_reg
        x_k_reg = key_reg
        x_q_cls = query_cls
        x_k_cls = key_cls
        x_v_reg = value
        N, B, C1 = x_q_reg.shape
        _, _, C2 = x_q_cls.shape
        _, _, C3 = x_v_reg.shape

        q_reg = self.q_reg(x_q_reg).reshape(N, B, self.num_heads, C1 // self.num_heads).permute(1, 2, 0, 3)
        k_reg = self.k_reg(x_k_reg).reshape(N, B, self.num_heads, C1 // self.num_heads).permute(1, 2, 0, 3)
        q_cls = self.q_cls(x_q_cls).reshape(N, B, self.num_heads, C2 // self.num_heads).permute(1, 2, 0, 3)
        k_cls = self.k_cls(x_k_cls).reshape(N, B, self.num_heads, C2 // self.num_heads).permute(1, 2, 0, 3)
        v_reg = self.v_reg(x_v_reg).reshape(N, B, self.num_heads, C3 // self.num_heads).permute(1, 2, 0, 3)

        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1))

        attn_cls = (q_cls @ k_cls.transpose(-2, -1))

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn = (attn_reg + attn_cls) / 2

        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C3)

        return x_reg.transpose(0, 1), attn


# class SEModule(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEModule, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels//16, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels//16, channels, bias=False),
#             nn.Sigmoid()
#         )
#         self.reduce = nn.Linear(channels, channels//2)
#
#     def forward(self, reg_feature, edge_feature):
#         q, b, c = reg_feature.shape
#         reg_feature = reg_feature.view(q, c)
#         edge_feature = edge_feature.view(q, c)
#         feature = torch.cat([reg_feature, edge_feature], dim=1)
#         feature_weight = self.fc(feature)
#         feature = feature * feature_weight
#         return self.reduce(feature).view(q, b, c)


class SEModule(nn.Module):
    """
        SENet channel attention module
    """

    def __init__(self, channels):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels * 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, reg_feature, edge_feature):
        q, b, c = reg_feature.shape
        feature = torch.stack([reg_feature, edge_feature], dim=3)
        feature = feature.view(q * c, 2)
        feature_weight = self.fc(feature).view(q, b, c, 2)
        return reg_feature * feature_weight[:, :, :, 0] + edge_feature * feature_weight[:, :, :, 1]


class SelfAttentionLayer(nn.Module):
    """
        Self attention Layer
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = PositionMHAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.CA = SEModule(2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_edge: Optional[Tensor] = None,
                     query_boxes: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt if query_edge is None else self.CA(tgt, query_edge), query_pos)
        tgt2 = \
            self.self_attn(q, k, value=tgt, q_boxes=None if query_boxes == [] or query_boxes is None else query_boxes,
                           k_boxes=None if query_boxes == [] or query_boxes is None else query_boxes,
                           attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    query_edge: Optional[Tensor] = None,
                    query_boxes: Optional[Tensor] = None):
        tgt2 = self.norm(tgt if query_edge is None else self.CA(tgt, query_edge))
        # tgt2 = tgt if query_edge is None else self.CA(tgt, query_edge)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = \
            self.self_attn(q, k, value=tgt2, q_boxes=None if query_boxes == [] or query_boxes is None else query_boxes,
                           k_boxes=None if query_boxes == [] or query_boxes is None else query_boxes,
                           attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_edge: Optional[Tensor] = None,
                query_boxes: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos, query_edge, query_boxes)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos, query_edge, query_boxes)


class DoubleSelfAttentionLayer(nn.Module):
    """
        Double branch Self attention Layer
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = DoubleMHAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_reg, tgt_cls, value):
        q_reg = k_reg = tgt_reg
        q_cls = k_cls = tgt_cls
        tgt = self.self_attn(q_reg, k_reg, q_cls, k_cls, value=value)[0]
        tgt = value + self.dropout(tgt)
        tgt = self.norm(tgt)

        return tgt


class CrossAttentionLayer(nn.Module):
    """
        Cross attention Layer
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="silu", normalize_before=False):
        super().__init__()
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MHAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """
        Feed-forward network layer
    """

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="silu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return F.silu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ReferringCrossAttentionLayer(nn.Module):
    """
        Referring cross-attention Layer
    """

    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.0,
            activation="relu",
            normalize_before=False
    ):
        super().__init__()
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = PositionMHAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.CA = SEModule(2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            indentify,
            tgt,
            memory,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
            edge=None,
            query_edge=None,
            boxes=None,
            query_boxes=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt if query_edge is None else self.CA(tgt, query_edge), query_pos),
            key=self.with_pos_embed(memory if edge is None else self.CA(memory, edge), pos),
            value=memory, q_boxes=None if query_boxes is None or query_boxes == [] else query_boxes,
            k_boxes=None if boxes is None or boxes == [] else boxes,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
            self,
            indentify,
            tgt,
            memory,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
            edge=None,
            query_edge=None,
            boxes=None,
            query_boxes=None
    ):
        tgt2 = self.norm(tgt if query_edge is None else self.CA(tgt, query_edge))
        # tgt2 = tgt if query_edge is None else self.CA(tgt, query_edge)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory if edge is None else self.CA(memory, edge), pos),
            value=memory, q_boxes=None if query_boxes is None or query_boxes == [] else query_boxes,
            k_boxes=None if boxes is None or boxes == [] else boxes,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
            self,
            indentify,
            tgt,
            memory,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
            edge=None,
            query_edge=None,
            boxes=None,
            query_boxes=None,
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, edge, query_edge, boxes, query_boxes)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, edge, query_edge, boxes, query_boxes)


class AwarePositionRegMatcher(torch.nn.Module):
    """
        Aware position Regression Matcher
        reduces accumulation of noisy localization during long-term feature alignment
        further refines feature matching and increases attention on matched objects
    """

    def __init__(
            self,
            hidden_channel=512,
            feedforward_channel=2048,
            num_head=8,
            decoder_layer_num=6,
            act='silu'
    ):
        super(AwarePositionRegMatcher, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_aware_cross_attention_layers = nn.ModuleList()
        # self.transformer_position_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    activation=act,
                    normalize_before=False,
                )
            )

            self.transformer_aware_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    activation=act,
                    normalize_before=False,
                )
            )

            # self.transformer_position_cross_attention_layers.append(
            #     ReferringCrossAttentionLayer(
            #         d_model=hidden_channel,
            #         nhead=num_head,
            #         dropout=0.0,
            #         activation=act,
            #         normalize_before=False,
            #     )
            # )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    activation=act,
                    normalize_before=False,
                )
            )
        self.absolute_position_embedding = nn.Linear(256, hidden_channel)
        self.edge_feature_embedding = nn.Linear(int(hidden_channel / 4), hidden_channel)
        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None
        self.last_frame_reg_embeds = None
        self.last_frame_cls_embeds = None
        self.last_frame_boxes = None
        self.last_frame_time_embedding = None
        self.last_frame_edge_embeds = None

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, features, features_reg, features_cls, features_edge, preds_per_frame, time_embedding,
                resume=False):
        """
            :param features: the object features output
            :param features_reg: the enhanced object regression feature output
            :param features_cls: the enhanced object classification feature output
            :param features_edge: the object surrounding regression feature output
            :param preds_per_frame: the number of feature selected predictions per frame
            :param time_embedding: the absolute time embedding
            :param resume: whether the first frame is the start of the video
            :return: output matched features
        """
        if features_edge.shape[-1] != features.shape[-1]:
            features_edge = self.edge_feature_embedding(features_edge)
        features = features[:, None, :]  # q, b, c
        features_reg = features_reg[:, None, :]
        features_cls = features_cls[:, None, :]
        features_edge = features_edge[:, None, :]
        # get the features corresponding to each frame
        features_list = []
        features_reg_list = []
        features_cls_list = []
        features_edge_list = []
        for i, pred in enumerate(preds_per_frame):
            features_list.append(features[:pred])
            features_reg_list.append(features_reg[:pred])
            features_cls_list.append(features_cls[:pred])
            features_edge_list.append(features_edge[:pred])
            features = features[pred:]
            features_reg = features_reg[pred:]
            features_cls = features_cls[pred:]
            features_edge = features_edge[pred:]
        n_frame = len(preds_per_frame)
        # get the absolute position embedding
        time_embedding = self.absolute_position_embedding(time_embedding)[:, None, :]
        outputs = []
        ret_indices = []

        # enhanced object features with frame-by-frame matching
        for i in range(n_frame):
            if preds_per_frame[i] == 0:
                if i == 0 and resume is False:
                    self.last_outputs = None
                    self.last_frame_embeds = None
                    self.last_frame_reg_embeds = None
                    self.last_frame_cls_embeds = None
                    self.last_frame_boxes = None
                    self.last_frame_time_embedding = None
                    self.last_frame_edge_embeds = None
                continue
            ms_output = []
            single_frame_embeds = features_list[i]  # q b c
            single_frame_reg_embeds = features_reg_list[i]
            single_frame_cls_embeds = features_cls_list[i]
            single_frame_edge_embeds = features_edge_list[i]
            single_frame_time_embedding = time_embedding[i].unsqueeze(0)
            # the first frame of a video
            if i == 0 and resume is False or self.last_outputs is None:
                self._clear_memory()
                self.last_frame_embeds = single_frame_embeds
                self.last_frame_reg_embeds = single_frame_reg_embeds
                self.last_frame_cls_embeds = single_frame_cls_embeds
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        # get the sequence indexes of the matching reference and current frames
                        ret_indices.append(self.double_match_embds(single_frame_reg_embeds, single_frame_reg_embeds,
                                                                   single_frame_cls_embeds, single_frame_cls_embeds)[1])
                        output = self.transformer_aware_cross_attention_layers[j](
                            single_frame_embeds, single_frame_embeds, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=single_frame_time_embedding, query_pos=single_frame_time_embedding,
                            edge=single_frame_edge_embeds, query_edge=single_frame_edge_embeds
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_aware_cross_attention_layers[j](
                            ms_output[-1], ms_output[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=single_frame_time_embedding, query_pos=single_frame_time_embedding,
                            edge=single_frame_edge_embeds, query_edge=single_frame_edge_embeds

                        )
                        ms_output.append(output)
            else:
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        # get the sequence indexes of the matching reference and current frames
                        indices = self.double_match_embds(self.last_frame_reg_embeds, single_frame_reg_embeds,
                                                          self.last_frame_cls_embeds, single_frame_cls_embeds)
                        # different number of objects predicted for reference and current frames
                        if len(self.last_frame_embeds) < len(single_frame_embeds):
                            indices_no_match = []
                            for i in range(len(single_frame_embeds)):
                                if i not in indices[1]:
                                    indices_no_match.append(i)
                            single_frame_no_match_features = single_frame_embeds[indices_no_match]
                            single_frame_no_match_reg_features = single_frame_reg_embeds[indices_no_match]
                            single_frame_no_match_cls_features = single_frame_cls_embeds[indices_no_match]
                            single_frame_edge_no_match_features = single_frame_edge_embeds[indices_no_match]
                            matched_features = torch.cat(
                                (single_frame_embeds[indices[1]], single_frame_no_match_features), dim=0)
                            matched_reg_features = torch.cat(
                                (single_frame_reg_embeds[indices[1]], single_frame_no_match_reg_features), dim=0)
                            matched_cls_features = torch.cat(
                                (single_frame_cls_embeds[indices[1]], single_frame_no_match_cls_features), dim=0)
                            matched_edge_features = torch.cat(
                                (single_frame_edge_embeds[indices[1]], single_frame_edge_no_match_features), dim=0)
                            last_features = torch.cat((self.last_outputs[-1], single_frame_no_match_features), dim=0)
                            last_edge_features = torch.cat(
                                (self.last_frame_edge_embeds, single_frame_edge_no_match_features), dim=0)
                            indices = list(indices)
                            indices[1] = np.append(indices[1], indices_no_match)
                        elif len(self.last_frame_embeds) > len(single_frame_embeds):
                            matched_features = single_frame_embeds[indices[1]]
                            matched_reg_features = single_frame_reg_embeds[indices[1]]
                            matched_cls_features = single_frame_cls_embeds[indices[1]]
                            matched_edge_features = single_frame_edge_embeds[indices[1]]
                            last_features = self.last_outputs[-1][indices[0]]
                            last_edge_features = self.last_frame_edge_embeds[indices[0]]
                        else:
                            matched_features = single_frame_embeds[indices[1]]
                            matched_reg_features = single_frame_reg_embeds[indices[1]]
                            matched_cls_features = single_frame_cls_embeds[indices[1]]
                            matched_edge_features = single_frame_edge_embeds[indices[1]]
                            last_features = self.last_outputs[-1]
                            last_edge_features = self.last_frame_edge_embeds
                        self.last_frame_embeds = matched_features
                        self.last_frame_reg_embeds = matched_reg_features
                        self.last_frame_cls_embeds = matched_cls_features
                        ret_indices.append(indices[1])
                        output = self.transformer_aware_cross_attention_layers[j](
                            matched_features, last_features, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=single_frame_time_embedding, query_pos=self.last_frame_time_embedding,
                            edge=single_frame_edge_embeds, query_edge=last_edge_features
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_aware_cross_attention_layers[j](
                            ms_output[-1], last_features, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=single_frame_time_embedding, query_pos=self.last_frame_time_embedding,
                            edge=single_frame_edge_embeds, query_edge=last_edge_features
                        )
                        ms_output.append(output)
                single_frame_edge_embeds = matched_edge_features
            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            # save last frame information
            self.last_outputs = ms_output
            self.last_frame_time_embedding = single_frame_time_embedding
            self.last_frame_edge_embeds = single_frame_edge_embeds
            outputs.append(ms_output[1:])
        # get the output of the original sort
        for i in range(len(outputs)):
            outputs[i] = outputs[i][:, np.argsort(ret_indices[i])]
        if len(outputs) > 0:
            outputs = torch.cat(outputs, dim=1)  # (l, t*q, b, c)
            outputs = self.decoder_norm(outputs).squeeze(2)
        else:
            outputs = None
        return outputs

    def match_embds(self, ref_embds, cur_embds):
        """
            Feature Matching
            compute the similarity between the current features and the reference features
            use the hungarian algorithm to match the object features between adjacent frames
        """
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(ref_embds, cur_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        # indices = linear_sum_assignment(C.transpose(0, 1))
        # indices = indices[1]
        indices = linear_sum_assignment(C)

        return indices

    def double_match_embds(self, ref_cls_embds, cur_cls_embds, ref_reg_embds, cur_reg_embds):
        """
            Dual-Branch Feature Matching
            compute the dual-branch similarity the global-enhanced classification and regression features
            use the hungarian algorithm to match the object features between adjacent frames
        """
        #  embeds (q, b, c)
        ref_cls_embds, cur_cls_embds = ref_cls_embds.detach()[:, 0, :], cur_cls_embds.detach()[:, 0, :]
        ref_cls_embds = ref_cls_embds / (ref_cls_embds.norm(dim=1)[:, None] + 1e-6)
        cur_cls_embds = cur_cls_embds / (cur_cls_embds.norm(dim=1)[:, None] + 1e-6)
        cls_cos_sim = torch.mm(ref_cls_embds, cur_cls_embds.transpose(0, 1))
        ref_reg_embds, cur_reg_embds = ref_reg_embds.detach()[:, 0, :], cur_reg_embds.detach()[:, 0, :]
        ref_reg_embds = ref_reg_embds / (ref_reg_embds.norm(dim=1)[:, None] + 1e-6)
        cur_reg_embds = cur_reg_embds / (cur_reg_embds.norm(dim=1)[:, None] + 1e-6)
        reg_cos_sim = torch.mm(ref_reg_embds, cur_reg_embds.transpose(0, 1))
        C = 1 - ((cls_cos_sim + reg_cos_sim) / 2)

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        # indices = linear_sum_assignment(C.transpose(0, 1))
        # indices = indices[1]
        # hungarian algorithm
        indices = linear_sum_assignment(C)

        return indices

    def double_match_embds_position(self, ref_cls_embds, cur_cls_embds, ref_reg_embds, cur_reg_embds, ref_boxes,
                                    cur_boxes):
        """
            Dual-Branch Feature Matching with Position
            compute the dual-branch similarity the global-enhanced classification and regression features
            besides compute the similarity between the current object positions and the reference object positions
            use the hungarian algorithm to match the object features between adjacent frames
        """
        #  embeds (q, b, c)
        ref_cls_embds, cur_cls_embds = ref_cls_embds.detach()[:, 0, :], cur_cls_embds.detach()[:, 0, :]
        ref_cls_embds = ref_cls_embds / (ref_cls_embds.norm(dim=1)[:, None] + 1e-6)
        cur_cls_embds = cur_cls_embds / (cur_cls_embds.norm(dim=1)[:, None] + 1e-6)
        cls_cos_sim = torch.mm(ref_cls_embds, cur_cls_embds.transpose(0, 1))
        ref_reg_embds, cur_reg_embds = ref_reg_embds.detach()[:, 0, :], cur_reg_embds.detach()[:, 0, :]
        ref_reg_embds = ref_reg_embds / (ref_reg_embds.norm(dim=1)[:, None] + 1e-6)
        cur_reg_embds = cur_reg_embds / (cur_reg_embds.norm(dim=1)[:, None] + 1e-6)
        cls_reg_sim = torch.mm(ref_reg_embds, cur_reg_embds.transpose(0, 1))
        if ref_boxes == [] or ref_boxes is None or cur_boxes == [] or cur_boxes is None:
            C = 1 - ((cls_cos_sim + cls_reg_sim) / 2)
        else:
            position_embedding = self.cal_position_embedding(ref_boxes.detach(), cur_boxes.detach())
            position_embedding = torch.sum(position_embedding, dim=1, keepdim=False)[0] / position_embedding.size(1)
            C = 1 - ((cls_cos_sim + cls_reg_sim) / 2) + (1 - position_embedding)

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        # indices = linear_sum_assignment(C.transpose(0, 1))
        # indices = indices[1]
        # hungarian algorithm
        indices = linear_sum_assignment(C)

        return indices

    def match_embds_position(self, ref_reg_embds, cur_reg_embds, ref_boxes, cur_boxes):
        """
            Feature Matching with Position
            compute the similarity between the current features and the reference features
            besides compute the similarity between the current object positions and the reference object positions
            use the hungarian algorithm to match the object features between adjacent frames
        """
        #  embeds (q, b, c)
        ref_reg_embds, cur_reg_embds = ref_reg_embds.detach()[:, 0, :], cur_reg_embds.detach()[:, 0, :]
        ref_reg_embds = ref_reg_embds / (ref_reg_embds.norm(dim=1)[:, None] + 1e-6)
        cur_reg_embds = cur_reg_embds / (cur_reg_embds.norm(dim=1)[:, None] + 1e-6)
        cls_reg_sim = torch.mm(ref_reg_embds, cur_reg_embds.transpose(0, 1))
        position_embedding = self.cal_position_embedding(ref_boxes.detach(), cur_boxes.detach())
        position_embedding = torch.sum(position_embedding, dim=1, keepdim=False)[0] / position_embedding.size(1)
        C = (1 - cls_reg_sim) + (1 - position_embedding)

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        # indices = linear_sum_assignment(C.transpose(0, 1))
        # indices = indices[1]
        indices = linear_sum_assignment(C)

        return indices

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        """
           extract relative position embedding
        """
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        """
            extract the detection boxes relative position information
        """
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def cal_position_embedding(self, rois1, rois2):
        """
            get relative embedding
            :param rois1: x1y1x2y2
            :param rois2: x1y1x2y2
            :param feat_dim: feature dim
        """
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding


class TaskAligned(torch.nn.Module):
    """
        multi-head cross-attention
        to align the classification and regression tasks
    """
    def __init__(
            self,
            hidden_channel=512,
            num_head=8,
            decoder_layer_num=6,
            act='silu'
    ):
        super(TaskAligned, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_cross_attention_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    activation=act,
                    normalize_before=False,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_channel)

    def forward(self, features_reg, features_obj, preds_per_frame):
        """
        :param features: the object features output
        :param resume: whether the first frame is the start of the video
        :return: output matched features
        """
        features_reg = features_reg[:, None, :]  # q, b, c
        features_obj = features_obj[:, None, :]
        features_reg_list = []
        features_obj_list = []
        for i, pred in enumerate(preds_per_frame):
            features_reg_list.append(features_reg[:pred])
            features_obj_list.append(features_obj[:pred])
            features_reg = features_reg[pred:]
            features_obj = features_obj[pred:]
        n_frame = len(preds_per_frame)

        outputs = []
        for i in range(n_frame):
            if preds_per_frame[i] == 0:
                continue
            single_frame_reg_embeds = features_reg_list[i]  # q b c
            single_frame_obj_embeds = features_obj_list[i]
            for j in range(self.num_layers):
                single_frame_obj_embeds = self.transformer_cross_attention_layers[j](
                    single_frame_obj_embeds, single_frame_reg_embeds)
            outputs.append(single_frame_obj_embeds)
        if len(outputs) > 0:
            outputs = torch.cat(outputs, dim=0)  # (t*q, b, c)
            outputs = self.decoder_norm(outputs).squeeze(1)
        else:
            outputs = None
        return outputs
