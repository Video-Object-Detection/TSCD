import copy

import torch
import torch.nn as nn
from .weight_init import trunc_normal_
from .losses import IOUloss
from torch.nn import functional as F
from matplotlib import pyplot as plt
from yolox.utils.box_op import (box_cxcywh_to_xyxy, generalized_box_iou, extract_position_matrix,
                                extract_position_embedding,
                                pure_position_embedding)
from yolox.utils import bboxes_iou
from yolox.data.datasets.vid import get_timing_signal_1d
from yolox.models.post_process import get_linking_mat
import time


def visual_attention(data):
    data = data.cpu()
    data = data.detach().numpy()

    plt.xlabel('x')
    plt.ylabel('score')
    plt.imshow(data)
    plt.show()


def get_position_embedding(rois1, rois2, feat_dim=64):
    """
    get relative embedding
    :param rois1: x1y1x2y2
    :param rois2: x1y1x2y2
    :param feat_dim: feature dim
    """
    # [num_rois, num_ref_rois, 4]
    position_matrix = extract_position_matrix(rois1, rois2)
    # [num_rois, num_ref_rois, 64]
    position_embedding = extract_position_embedding(position_matrix, feat_dim=feat_dim)
    # [64, num_rois, num_ref_rois]
    position_embedding = position_embedding.permute(2, 0, 1)
    # [1, 64, num_rois, num_ref_rois]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding


class SelfAttentionLocal(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, attn_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.use_time_emd = kwargs.get('use_time_emd', False)
        self.use_loc_emb = kwargs.get('use_loc_emd', True)
        self.loc_fuse_type = kwargs.get('loc_fuse_type', 'add')
        self.use_qkv = kwargs.get('use_qkv', True)
        self.locf_dim = kwargs.get('locf_dim', 64)
        self.loc_emd_dim = kwargs.get('loc_emd_dim', 64)
        self.pure_pos_emb = kwargs.get('pure_pos_emb', False)
        self.loc_conf = kwargs.get('loc_conf', False)
        self.iou_base = kwargs.get('iou_base', False)
        self.iou_thr = kwargs.get('iou_thr', 0.5)
        self.reconf = kwargs.get('reconf', False)
        self.iou_window = kwargs.get('iou_window', 0)
        if self.iou_base:
            self.use_time_emd = False
            self.use_loc_emb = False
            self.pure_pos_emb = False

        if self.reconf:
            self.qk = nn.Linear(dim * 2, dim * 4, bias=bias)
            self.v_cls = nn.Linear(dim, dim, bias=bias)
            self.v_reg = nn.Linear(dim, dim, bias=bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=bias)
            # self.qk = nn.Linear(dim * 2, dim * 4, bias=bias)
            # self.v_cls = nn.Linear(dim, dim, bias=bias)

        if self.use_loc_emb:
            if self.pure_pos_emb:
                self.loc2feature = nn.Linear(4, dim, bias=False)
                self.loc_fuse_type = 'identity'
            else:
                self.loc2feature = nn.Conv2d(self.locf_dim, self.num_heads, kernel_size=1, stride=1, padding=0)
                # init the loc2feature
                trunc_normal_(self.loc2feature.weight, std=0.01)
                nn.init.constant_(self.loc2feature.bias, 0)
                self.locAct = nn.ReLU(inplace=True)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, x_reg, locs, **kwargs):
        B, N, C = x.shape
        L, G, P = kwargs.get('lframe'), kwargs.get('gframe'), kwargs.get('afternum')

        assert B == 1, 'only support video batch size 1 currently'
        if locs != None:
            LF, P = locs.shape[0], locs.shape[1]
            locs = locs.view(-1, 4)

        if self.use_loc_emb and not self.pure_pos_emb:
            loc_emd = get_position_embedding(locs, locs, feat_dim=self.loc_emd_dim).type_as(x)  # 1, 64, N, N
            if self.use_time_emd:
                time_emd = get_timing_signal_1d(torch.arange(0, LF), self.locf_dim).type_as(x)  # LF, 64
                time_emd = time_emd.unsqueeze(1).repeat(P, N, 1).permute(2, 0, 1).unsqueeze(0)  # 1, 64, N, N
                loc_time_emb = loc_emd + time_emd
            else:
                loc_time_emb = loc_emd
            attn_lt = self.locAct(self.loc2feature(loc_time_emb))  #
            fg_score = kwargs.get('fg_score', None)
            if self.loc_conf and fg_score is not None:
                fg_score = fg_score > 0.001
                fg_score = fg_score.view(1, -1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, N, 1)
                fg_score = fg_score.type_as(x)
                attn_lt = attn_lt * fg_score
        elif self.pure_pos_emb:
            pure_loc_features = pure_position_embedding(locs, kwargs.get('width'), kwargs.get('height')).type_as(x)
            pure_loc_features = self.loc2feature(pure_loc_features)  # B*N,C
            pure_loc_features = pure_loc_features.view(B, N, C)
            if self.use_time_emd:
                time_emd = get_timing_signal_1d(torch.arange(0, LF), C).type_as(x)  # LF, C
                time_emd = time_emd.unsqueeze(0).repeat(B, P, 1)  # B, N, C
                pure_loc_features = pure_loc_features + time_emd
            x = x + pure_loc_features.reshape(B, N, -1)
        elif self.iou_base:
            if self.iou_window != 0:
                iou_masks = torch.zeros((N, N))
                for i in range(L):
                    lower = max(i - self.iou_window, 0)
                    upper = min(i + self.iou_window, L)
                    iou_masks[lower * P:upper * P, i * P:(i + 1) * P] = 1
                iou_masks = iou_masks.type_as(x)
            else:
                iou_masks = 1
                # set the
            iou_mat = bboxes_iou(locs, locs)  # N,N
            iou_mat = (iou_mat > 0.0) * iou_masks

        if self.reconf:
            qk = self.qk(torch.cat([x, x_reg], dim=-1))
            qk = qk.reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk[0], qk[1]
            v_cls = self.v_cls(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v_loc = self.v_reg(x_reg).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v_cls = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, C
            # qk = self.qk(torch.cat([x, x_reg], dim=-1))
            # qk = qk.reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q, k = qk[0], qk[1]
            # v_cls = self.v_cls(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, M, N
        cls_score = kwargs.get('cls_score', None)
        if self.loc_conf and cls_score is not None:
            cls_score = cls_score.view(1, -1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, N, 1)
            cls_score = cls_score.type_as(x)
            attn = attn * cls_score

        if self.loc_fuse_type == 'add' and not self.iou_base:
            attn = attn + (attn_lt + 1e-6).log()
        elif self.loc_fuse_type == 'dot' and not self.iou_base:
            attn = attn * (attn_lt + 1e-6).log()
        elif self.loc_fuse_type == 'identity' or self.iou_base:
            attn = attn
        else:
            raise NotImplementedError
        attn = attn.softmax(dim=-1)
        if self.iou_base:
            attn = attn * iou_mat.type_as(x)
            attn = attn / (torch.sum(attn, dim=-1, keepdim=True))
        attn = self.attn_drop(attn)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        if self.reconf:
            x_reg = (attn @ v_loc).transpose(1, 2).reshape(B, N, C)
            return x, x_reg
        return x


class FFN(nn.Module):
    """
        Feed-Forward Neural Network
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, dropout=0., attn_drop=0., drop_path=0., **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttentionLocal(dim, num_heads=num_heads, bias=qkv_bias, attn_drop=attn_drop, **kwargs)
        self.drop_path = nn.Identity()
        self.use_ffn = kwargs.get('use_ffn', True)
        self.reconf = kwargs.get('reconf', False)
        self.norm3 = nn.LayerNorm(dim)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = FFN(dim, int(dim * mlp_ratio), dropout=dropout)
            if self.reconf:
                self.norm4 = nn.LayerNorm(dim)
                self.mlp_conf = FFN(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, x_reg, locs, **kwargs):
        if self.reconf:
            x_cls, x_reg_ = self.attn(self.norm1(x), self.norm3(x_reg), locs, **kwargs)
            x_reg = x_reg + self.drop_path(x_reg_)
            x_cls = x_cls + self.drop_path(x)
            if self.use_ffn:
                x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
                x_reg = x_reg + self.drop_path(self.mlp_conf(self.norm4(x_reg)))
            return x_cls, x_reg
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm3(x_reg), locs, **kwargs))
            if self.use_ffn:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, x_reg


class Attention_aware_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # the scaled dot-product attention
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,
                return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False, **kwargs):
        B, N, C = x_cls.shape
        # multi-head self-attention
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

        # avoid feature homogenization
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        # similarity matrix
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        # remove ave and conf guide in the reg branch, modified in 2023.12.5
        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale  # * fg_score * fg_score_mask

        if kwargs.get('local_mask', False):
            lframe, gframe, P = kwargs.get('lframe'), kwargs.get('gframe'), kwargs.get('afternum')
            local_mask_branch = kwargs.get('local_mask_branch')
            if 'cls' in local_mask_branch:
                attn_cls[:, :, 0:lframe * P, 0:lframe * P] = -1e4
            if 'reg' in local_mask_branch:
                attn_reg[:, :, 0:lframe * P, 0:lframe * P] = -1e4

        # double branch
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        #
        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)
        x_ori_reg = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)
        x_reg = torch.cat([x_reg, x_ori_reg], dim=-1)

        # preserving feature diversity
        if ave:
            conf_sim_thresh = kwargs.get('conf_sim_thresh', 0.99)
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            # remove ave and conf guide in the reg branch, modified in 2023.12.5
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)
            if use_mask:
                sim_mask = sim_mask * cls_score_mask[0, 0, :, :] * fg_score_mask[0, 0, :, :]
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            obj_mask = obj_mask * sim_round2 / (torch.sum(obj_mask * sim_round2, dim=-1, keepdim=True))
            return x_cls, x_reg, sim_round2, obj_mask
        else:
            return x_cls, x_reg, None, None


class SEModule(nn.Module):
    """
        SENet channel attention module
    """
    def __init__(self, channels):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels*16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels*16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, raw_feature, enhance_feature):
        b, q, c = raw_feature.shape
        feature = torch.stack([raw_feature, enhance_feature], dim=3)
        feature = feature.view(q*c, 2)
        feature_weight = self.fc(feature).view(b, q, c, 2)
        return raw_feature*feature_weight[:, :, :, 0] + enhance_feature*feature_weight[:, :, :, 1]


class Attention_mca_aware_g2l(nn.Module):
    """
        Multi-Head Cross-Attention
        the queries denote the feature selection results of the key frame,
        the keys and values denote the feature selection results of the global frame.
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
        Adaptive fusion of key features of the classification branch
        and edge features of the regression branch via channel attention to enhance feature match accuracy.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # the scaled dot-product attention
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.reconf = reconf

        self.q_cls_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_cls_enhanced = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_cls = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_reg_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_reg_enhanced = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_reg = nn.Linear(dim, dim, bias=qkv_bias)

        # SENet channel attention
        self.key_enhance = SEModule(2)
        self.edge_enhance = SEModule(2)

        self.attn_drop = nn.Dropout(attn_drop)

        self.linear = nn.Linear(2 * dim, 2 * dim)
        if self.reconf:
            self.linear_reg = nn.Linear(2 * dim, 2 * dim)

    def find_similar_round2(self, features, support_cls, ave_mask, feature_reg, support_reg, mask_reg, fg_score=None):
        """
            preserving feature diversity
        """
        key_feature = features[0]
        support_feature = support_cls[0]
        if not self.training:
            ave_mask = ave_mask.to(features.dtype)
        soft_sim_feature = (ave_mask @ support_feature)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        if self.reconf:
            mask_reg = mask_reg.to(features.dtype)
            key_feature_reg = feature_reg[0]
            support_feature_reg = support_reg[0]
            soft_sim_feature_reg = (mask_reg @ support_feature_reg)
            reg_feature = torch.cat([soft_sim_feature_reg, key_feature_reg], dim=-1)
        else:
            reg_feature = None
        return cls_feature, reg_feature

    def forward(self, x_cls, x_reg, x_key, x_edge, cls_score=None, fg_score=None,
                return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False, **kwargs):
        # get the feature selected local feature
        local_preds_num = kwargs.get('local_preds_num')
        x_cls_local = x_cls[:, :local_preds_num, :]
        x_key_local = x_key[:, :local_preds_num, :]
        # enhance the key feature for classification branch
        x_cls_local = self.key_enhance(x_cls_local, x_key_local)
        x_cls_enhanced = self.key_enhance(x_cls, x_key)
        x_reg_local = x_reg[:, :local_preds_num, :]
        x_edge_local = x_edge[:, :local_preds_num, :]
        # enhance the edge feature for regression branch
        x_reg_local = self.edge_enhance(x_reg_local, x_edge_local)
        x_reg_enhanced = self.edge_enhance(x_reg, x_edge)

        B, N1, C = x_cls_local.shape
        _, N2, _ = x_cls.shape

        # multi-head cross-attention
        q_cls_local = (self.q_cls_local(x_cls_local).reshape(B, N1, 1, self.num_heads, C // self.num_heads).
                       permute(2, 0, 3, 1, 4))
        k_cls_enhanced = self.k_cls_enhanced(x_cls_enhanced).reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v_cls = self.v_cls(x_cls).reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_reg_local = (self.q_reg_local(x_reg_local).reshape(B, N1, 1, self.num_heads, C // self.num_heads).
                       permute(2, 0, 3, 1, 4))
        k_reg_enhanced = self.k_reg_enhanced(x_reg_enhanced).reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v_reg = self.v_reg(x_reg).reshape(B, N2, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = q_cls_local[0], k_cls_enhanced[0], v_cls[0]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = q_reg_local[0], k_reg_enhanced[0], v_reg[0]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

        # avoid feature homogenization during aggregation
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N1, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N1, 1)

        # similarity matrix
        attn_cls_raw = v_cls_normed[:, :, :N1, :] @ v_cls_normed.transpose(-2, -1)
        attn_reg_raw = v_reg_normed[:, :, :N1, :] @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        # remove ave and conf guide in the reg branch, modified in 2023.12.5
        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale  # * fg_score * fg_score_mask

        if kwargs.get('local_mask', False):
            lframe, gframe, P = kwargs.get('lframe'), kwargs.get('gframe'), kwargs.get('afternum')
            local_mask_branch = kwargs.get('local_mask_branch')
            if 'cls' in local_mask_branch:
                attn_cls[:, :, 0:lframe * P, 0:lframe * P] = -1e4
            if 'reg' in local_mask_branch:
                attn_reg[:, :, 0:lframe * P, 0:lframe * P] = -1e4

        # double branch
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N1, C)
        x_ori = v_cls[:, :, :N1, :].permute(0, 2, 1, 3).reshape(B, N1, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        #
        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N1, C)
        x_ori_reg = v_reg[:, :, :N1, :].permute(0, 2, 1, 3).reshape(B, N1, C)
        x_reg = torch.cat([x_reg, x_ori_reg], dim=-1)

        x_cls = self.linear(x_cls)  #
        if self.reconf:
            x_reg = self.linear_reg(x_reg)

        # preserving feature diversity
        if ave:
            conf_sim_thresh = kwargs.get('conf_sim_thresh', 0.99)
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            # remove ave and conf guide in the reg branch, modified in 2023.12.5
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)

            if use_mask:
                sim_mask = sim_mask * cls_score_mask[0, 0, :, :] * fg_score_mask[0, 0, :, :]
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            obj_mask = obj_mask * sim_round2 / (torch.sum(obj_mask * sim_round2, dim=-1, keepdim=True))

            x_cls, x_reg = self.find_similar_round2(x_cls, v_cls.permute(0, 2, 1, 3).reshape(B, N2, C), sim_round2,
                                                    x_reg, v_reg.permute(0, 2, 1, 3).reshape(B, N2, C), obj_mask)

        return x_cls, x_reg


class Attention_mca_g2l(nn.Module):
    """
        Multi-Head Cross-Attention
        the queries denote the feature selection results of the key frame,
        the keys and values denote the feature selection results of the global frame.
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # the scaled dot-product attention
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.reconf = reconf

        self.q_cls_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_cls = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_reg_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_reg = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.linear = nn.Linear(2 * dim, 2 * dim)
        if self.reconf:
            self.linear_reg = nn.Linear(2 * dim, 2 * dim)

    def find_similar_round2(self, features, support_cls, ave_mask, feature_reg, support_reg, mask_reg, fg_score=None):
        """
            preserving feature diversity
        """
        key_feature = features[0]
        support_feature = support_cls[0]
        if not self.training:
            ave_mask = ave_mask.to(features.dtype)
        soft_sim_feature = (ave_mask @ support_feature)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        if self.reconf:
            mask_reg = mask_reg.to(features.dtype)
            key_feature_reg = feature_reg[0]
            support_feature_reg = support_reg[0]
            soft_sim_feature_reg = (mask_reg @ support_feature_reg)
            reg_feature = torch.cat([soft_sim_feature_reg, key_feature_reg], dim=-1)
        else:
            reg_feature = None
        return cls_feature, reg_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,
                return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False, **kwargs):
        # get the local prediction feature numbers
        local_preds_num = kwargs.get('local_preds_num')
        x_cls_local = x_cls[:, :local_preds_num, :]
        x_reg_local = x_reg[:, :local_preds_num, :]

        B, N1, C = x_cls_local.shape
        _, N2, _ = x_cls.shape

        # for each cross-attention head, queries, keys and values perform independent linear projections
        q_cls_local = (self.q_cls_local(x_cls_local).reshape(B, N1, 1, self.num_heads, C // self.num_heads).
                       permute(2, 0, 3, 1, 4))
        kv_cls = self.kv_cls(x_cls).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_reg_local = (self.q_reg_local(x_reg_local).reshape(B, N1, 1, self.num_heads, C // self.num_heads).
                       permute(2, 0, 3, 1, 4))
        kv_reg = self.kv_reg(x_reg).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q_cls, k_cls, v_cls = q_cls_local[0], kv_cls[0], kv_cls[1]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = q_reg_local[0], kv_reg[0], kv_reg[1]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

        # avoid feature homogenization
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N1, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N1, 1)

        # similarity matrix
        attn_cls_raw = v_cls_normed[:, :, :N1, :] @ v_cls_normed.transpose(-2, -1)
        attn_reg_raw = v_reg_normed[:, :, :N1, :] @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        # multi-head cross-attention
        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        # remove ave and conf guide in the reg branch, modified in 2023.12.5
        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale  # * fg_score * fg_score_mask

        if kwargs.get('local_mask', False):
            lframe, gframe, P = kwargs.get('lframe'), kwargs.get('gframe'), kwargs.get('afternum')
            local_mask_branch = kwargs.get('local_mask_branch')
            if 'cls' in local_mask_branch:
                attn_cls[:, :, 0:lframe * P, 0:lframe * P] = -1e4
            if 'reg' in local_mask_branch:
                attn_reg[:, :, 0:lframe * P, 0:lframe * P] = -1e4

        # utilize both semantic similarity of classification and regression branches
        # to improve the accuracy of feature aggregation
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N1, C)
        x_ori = v_cls[:, :, :N1, :].permute(0, 2, 1, 3).reshape(B, N1, C)
        x_cls = torch.cat([x, x_ori], dim=-1)

        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N1, C)
        x_ori_reg = v_reg[:, :, :N1, :].permute(0, 2, 1, 3).reshape(B, N1, C)
        x_reg = torch.cat([x_reg, x_ori_reg], dim=-1)

        x_cls = self.linear(x_cls)  #
        if self.reconf:
            x_reg = self.linear_reg(x_reg)

        # preserving feature diversity
        if ave:
            conf_sim_thresh = kwargs.get('conf_sim_thresh', 0.99)
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            # remove ave and conf guide in the reg branch, modified in 2023.12.5
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)

            if use_mask:
                sim_mask = sim_mask * cls_score_mask[0, 0, :, :] * fg_score_mask[0, 0, :, :]
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            obj_mask = obj_mask * sim_round2 / (torch.sum(obj_mask * sim_round2, dim=-1, keepdim=True))

            x_cls, x_reg = self.find_similar_round2(x_cls, v_cls.permute(0, 2, 1, 3).reshape(B, N2, C), sim_round2,
                                                    x_reg, v_reg.permute(0, 2, 1, 3).reshape(B, N2, C), obj_mask)

        return x_cls, x_reg


class Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,
                return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False, **kwargs):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        # remove ave and conf guide in the reg branch, modified in 2023.12.5
        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale  # * fg_score * fg_score_mask

        if kwargs.get('local_mask', False):
            lframe, gframe, P = kwargs.get('lframe'), kwargs.get('gframe'), kwargs.get('afternum')
            local_mask_branch = kwargs.get('local_mask_branch')
            if 'cls' in local_mask_branch:
                attn_cls[:, :, 0:lframe * P, 0:lframe * P] = -1e4
            if 'reg' in local_mask_branch:
                attn_reg[:, :, 0:lframe * P, 0:lframe * P] = -1e4

        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        #
        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)
        x_ori_reg = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)
        x_reg = torch.cat([x_reg, x_ori_reg], dim=-1)

        if ave:
            conf_sim_thresh = kwargs.get('conf_sim_thresh', 0.99)
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            # print(torch.max(attn_cls_raw), torch.min(attn_cls_raw))
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
            # print(torch.max(attn_reg_raw), torch.min(attn_reg_raw))
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            # print(torch.sum(sim_mask))
            # remove ave and conf guide in the reg branch, modified in 2023.12.5
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)
            # print(torch.sum(obj_mask))
            if use_mask:
                sim_mask = sim_mask * cls_score_mask[0, 0, :, :] * fg_score_mask[0, 0, :, :]
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            obj_mask = obj_mask * sim_round2 / (torch.sum(obj_mask * sim_round2, dim=-1, keepdim=True))
            return x_cls, x_reg, sim_round2, obj_mask
        else:
            return x_cls, x_reg, None, None


class Attention_msa_visual(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = 30  # scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, img=None, pred=None):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score  # * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_cls_raw * 25).softmax(
            dim=-1)  # attn_cls#(attn_reg + attn_cls) / 2 #attn_reg#(attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)

        ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
        zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

        attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
        sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
        from yolox.models.post_process import visual_sim
        attn_total = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads
        visual_sim(attn_total, img, 30, pred, attn_cls_raw)
        return x_cls, None, sim_round2


class Attention_msa_online(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            return x_cls, None, sim_round2
        else:
            return x_cls


class LocalAggregation(nn.Module):
    def __init__(self, dim, heads, bias=False, attn_drop=0., blocks=1, **kwargs):
        super().__init__()
        self.blocks = blocks
        self.transBlocks = nn.ModuleList()
        for i in range(blocks):
            self.transBlocks.append(TransformerBlock(dim, heads, qkv_bias=bias, attn_drop=attn_drop, **kwargs))

    def forward(self, x, x_reg, locs=None, **kwargs):
        for i in range(self.blocks):
            x, x_reg = self.transBlocks[i](x, x_reg, locs, **kwargs)
        return x, x_reg


class MSA_aware_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        self.msa = Attention_aware_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)
        if reconf:
            self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear2_obj = nn.Linear(4 * dim, out_dim)

    def find_similar_round2(self, features, ave_mask, feature_obj, mask_obj, fg_score=None):
        """
            preserving feature diversity
        """
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            ave_mask = ave_mask.to(features.dtype)
        soft_sim_feature = (ave_mask @ support_feature)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        if self.reconf:
            mask_obj = mask_obj.to(features.dtype)
            key_feature_obj = feature_obj[0]
            support_feature_obj = feature_obj[0]
            soft_sim_feature_obj = (mask_obj @ support_feature_obj)
            obj_feature = torch.cat([soft_sim_feature_obj, key_feature_obj], dim=-1)
        else:
            obj_feature = None
        return cls_feature, obj_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        # multi-head self-attention
        trans_cls, trans_obj, ave_mask, obj_mask = self.msa(x_cls, x_reg, cls_score, fg_score,
                                                            sim_thresh=sim_thresh, ave=ave,
                                                            use_mask=use_mask, **kwargs)

        trans_cls = self.linear1(trans_cls)  #
        if self.reconf:
            trans_obj = self.linear1_obj(trans_obj)
        trans_cls, trans_obj = self.find_similar_round2(trans_cls, ave_mask, trans_obj, obj_mask)
        trans_cls = self.linear2(trans_cls)
        if self.reconf:
            trans_obj = self.linear2_obj(trans_obj)
        return trans_cls, trans_obj


class MCA_tscd_g2l_cls(nn.Module):
    """
        Multi-Head Cross-Attention for classification branch
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
        In addition, local features complement each other.
    """
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        # global to local multi-head cross-attention
        self.mca = Attention_mca_g2l(dim, num_heads, qkv_bias, attn_drop, scale=scale, reconf=reconf)
        # self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear = nn.Linear(3 * dim, out_dim)
        if reconf:
            # self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear_obj = nn.Linear(3 * dim, out_dim)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        # get the feature selected local prediction numbers
        preds_per_frame = kwargs.get('preds_per_frame')
        lframe = kwargs.get('lframe')
        local_preds_num = 0
        for i in range(lframe):
            local_preds_num += preds_per_frame[i]
        kwargs.update({'local_preds_num': local_preds_num})
        # multi-head cross-attention
        trans_cls, trans_obj = self.mca(x_cls, x_reg, cls_score, fg_score,
                                        sim_thresh=sim_thresh, ave=ave,
                                        use_mask=use_mask, **kwargs)
        trans_cls = self.linear(trans_cls)
        if self.reconf:
            trans_obj = self.linear_obj(trans_obj)
        return trans_cls, trans_obj


class MCA_tscd_aware_g2l_cls(nn.Module):
    """
        Multi-Head Cross-Attention for classification branch
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
        In addition, local features complement each other.
        Adaptive fusion of key features of the classification branch
        and edge features of the regression branch via channel attention to enhance feature match accuracy.
    """
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        self.mca = Attention_mca_aware_g2l(dim, num_heads, qkv_bias, attn_drop, scale=scale, reconf=reconf)
        # self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear = nn.Linear(3 * dim, out_dim)
        if reconf:
            # self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear_obj = nn.Linear(3 * dim, out_dim)

    def forward(self, x_cls, x_reg, x_key, x_edge, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        # get the feature selected local prediction numbers
        preds_per_frame = kwargs.get('preds_per_frame')
        lframe = kwargs.get('lframe')
        local_preds_num = 0
        for i in range(lframe):
            local_preds_num += preds_per_frame[i]
        kwargs.update({'local_preds_num': local_preds_num})
        # aware multi-head cross-attention
        trans_cls, trans_obj = self.mca(x_cls, x_reg, x_key, x_edge, cls_score, fg_score,
                                        sim_thresh=sim_thresh, ave=ave,
                                        use_mask=use_mask, **kwargs)

        trans_cls = self.linear(trans_cls)
        if self.reconf:
            trans_obj = self.linear_obj(trans_obj)
        return trans_cls, trans_obj


class MCA_tscd_g2l_reg(nn.Module):
    """
        Multi-Head Cross-Attention for regression branch
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
        Avoid feature mixing between adjacent frames to get accurate position regression.
        Frame-by-frame enhances the local feature.
    """
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        # global to local multi-head cross-attention
        self.mca = Attention_mca_g2l(dim, num_heads, qkv_bias, attn_drop, scale=scale, reconf=reconf)
        # self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear = nn.Linear(3 * dim, out_dim)
        if reconf:
            # self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear_obj = nn.Linear(3 * dim, out_dim)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        # get the feature selected local prediction numbers
        preds_per_frame = kwargs.get('preds_per_frame')
        lframe = kwargs.get('lframe')
        local_preds_num = 0
        for i in range(lframe):
            local_preds_num += preds_per_frame[i]
        x_cls_global = x_cls[:, local_preds_num:, :]
        x_reg_global = x_reg[:, local_preds_num:, :]
        cls_score_global = cls_score[local_preds_num:]
        fg_score_global = fg_score[local_preds_num:]
        # frame-by-frame enhances the local features
        start = 0
        trans_cls_local = []
        trans_obj_local = []
        for i in range(lframe):
            x_cls_per_frame = torch.cat((x_cls[:, start:start + preds_per_frame[i], :], x_cls_global), dim=1)
            x_reg_per_frame = torch.cat((x_reg[:, start:start + preds_per_frame[i], :], x_reg_global), dim=1)
            cls_score_per_frame = torch.cat((cls_score[start:start + preds_per_frame[i]], cls_score_global), dim=0)
            fg_score_per_frame = torch.cat((fg_score[start:start + preds_per_frame[i]], fg_score_global), dim=0)
            kwargs.update({'local_preds_num': preds_per_frame[i]})
            trans_cls, trans_obj = self.mca(x_cls_per_frame, x_reg_per_frame, cls_score_per_frame, fg_score_per_frame,
                                            sim_thresh=sim_thresh, ave=ave,
                                            use_mask=use_mask, **kwargs)
            trans_cls_local.append(trans_cls)
            trans_obj_local.append(trans_obj)
            start += preds_per_frame[i]

        trans_cls = torch.cat(trans_cls_local, dim=0)
        trans_obj = torch.cat(trans_obj_local, dim=0)

        trans_cls = self.linear(trans_cls)
        if self.reconf:
            trans_obj = self.linear_obj(trans_obj)
        return trans_cls, trans_obj


class MCA_tscd_aware_g2l_reg(nn.Module):
    """
        Multi-Head Cross-Attention for regression branch
        The features in the global frames are aligned and complemented to the keyframes by multi-head cross-attention.
        Avoid feature mixing between adjacent frames to get accurate position regression.
        Frame-by-frame enhances the local feature.
        Adaptive fusion of key features of the classification branch
        and edge features of the regression branch via channel attention to enhance feature match accuracy.
    """
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        # aware multi-head cross-attention
        self.mca = Attention_mca_aware_g2l(dim, num_heads, qkv_bias, attn_drop, scale=scale, reconf=reconf)
        # self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear = nn.Linear(3 * dim, out_dim)
        if reconf:
            # self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear_obj = nn.Linear(3 * dim, out_dim)

    def forward(self, x_cls, x_reg, x_key, x_edge, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        # get the feature selected local prediction numbers
        preds_per_frame = kwargs.get('preds_per_frame')
        lframe = kwargs.get('lframe')
        local_preds_num = 0
        for i in range(lframe):
            local_preds_num += preds_per_frame[i]
        x_cls_global = x_cls[:, local_preds_num:, :]
        x_reg_global = x_reg[:, local_preds_num:, :]
        x_key_global = x_key[:, local_preds_num:, :]
        x_edge_global = x_edge[:, local_preds_num:, :]
        cls_score_global = cls_score[local_preds_num:]
        fg_score_global = fg_score[local_preds_num:]
        # frame-by-frame enhances the local features
        start = 0
        trans_cls_local = []
        trans_obj_local = []
        for i in range(lframe):
            x_cls_per_frame = torch.cat((x_cls[:, start:start + preds_per_frame[i], :], x_cls_global), dim=1)
            x_reg_per_frame = torch.cat((x_reg[:, start:start + preds_per_frame[i], :], x_reg_global), dim=1)
            x_key_per_frame = torch.cat((x_key[:, start:start + preds_per_frame[i], :], x_key_global), dim=1)
            x_edge_per_frame = torch.cat((x_edge[:, start:start + preds_per_frame[i], :], x_edge_global), dim=1)
            cls_score_per_frame = torch.cat((cls_score[start:start + preds_per_frame[i]], cls_score_global), dim=0)
            fg_score_per_frame = torch.cat((fg_score[start:start + preds_per_frame[i]], fg_score_global), dim=0)
            kwargs.update({'local_preds_num': preds_per_frame[i]})
            trans_cls, trans_obj = self.msa(x_cls_per_frame, x_reg_per_frame, x_key_per_frame, x_edge_per_frame, cls_score_per_frame, fg_score_per_frame,
                                            sim_thresh=sim_thresh, ave=ave,
                                            use_mask=use_mask, **kwargs)
            trans_cls_local.append(trans_cls)
            trans_obj_local.append(trans_obj)
            start += preds_per_frame[i]

        trans_cls = torch.cat(trans_cls_local, dim=0)
        trans_obj = torch.cat(trans_obj_local, dim=0)

        trans_cls = self.linear(trans_cls)
        if self.reconf:
            trans_obj = self.linear_obj(trans_obj)
        return trans_cls, trans_obj


class MSA_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)
        if reconf:
            self.linear1_obj = nn.Linear(2 * dim, 2 * dim)
            self.linear2_obj = nn.Linear(4 * dim, out_dim)

    def find_similar_round2(self, features, ave_mask, feature_obj, mask_obj, fg_score=None):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            ave_mask = ave_mask.to(features.dtype)
        soft_sim_feature = (ave_mask @ support_feature)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        if self.reconf:
            mask_obj = mask_obj.to(features.dtype)
            key_feature_obj = feature_obj[0]
            support_feature_obj = feature_obj[0]
            soft_sim_feature_obj = (mask_obj @ support_feature_obj)
            obj_feature = torch.cat([soft_sim_feature_obj, key_feature_obj], dim=-1)
        else:
            obj_feature = None
        return cls_feature, obj_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True,
                use_mask=False, **kwargs):
        trans_cls, trans_obj, ave_mask, obj_mask = self.msa(x_cls, x_reg, cls_score, fg_score,
                                                            sim_thresh=sim_thresh, ave=ave,
                                                            use_mask=use_mask, **kwargs)

        trans_cls = self.linear1(trans_cls)  #
        if self.reconf:
            trans_obj = self.linear1_obj(trans_obj)
        trans_cls, trans_obj = self.find_similar_round2(trans_cls, ave_mask, trans_obj, obj_mask)
        trans_cls = self.linear2(trans_cls)
        if self.reconf:
            trans_obj = self.linear2_obj(trans_obj)
        return trans_cls, trans_obj


class MSA_yolov_visual(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Attention_msa_visual(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def ave_pooling_over_ref(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (
                sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, img=None, pred=None):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, img, pred)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa, sim_round2)
        out = self.linear2(ave)
        return out


class MSA_yolov_online(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Attention_msa_online(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def ave_pooling_over_ref(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (
                sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)

        return cls_feature

    def compute_geo_sim(self, key_preds, ref_preds):
        key_boxes = key_preds[:, :4]
        ref_boxes = ref_preds[:, :4]
        cost_giou, iou = generalized_box_iou(key_boxes.to(torch.float32), ref_boxes.to(torch.float32))

        return iou.to(torch.float16)

    def local_agg(self, features, local_results, boxes, cls_score, fg_score):
        local_features = local_results['msa']
        local_features_n = local_features / torch.norm(local_features, dim=-1, keepdim=True)
        features_n = features / torch.norm(features, dim=-1, keepdim=True)
        cos_sim = features_n @ local_features_n.transpose(0, 1)

        geo_sim = self.compute_geo_sim(boxes, local_results['boxes'])
        N = local_results['cls_scores'].shape[0]
        M = cls_score.shape[0]
        pre_scores = cls_score * fg_score
        pre_scores = torch.reshape(pre_scores, [-1, 1]).repeat(1, N)
        other_scores = local_results['cls_scores'] * local_results['reg_scores']
        other_scores = torch.reshape(other_scores, [1, -1]).repeat(M, 1)
        ones_matrix = torch.ones([M, N]).to('cuda')
        zero_matrix = torch.zeros([M, N]).to('cuda')
        thresh_map = torch.where(other_scores - pre_scores > -0.3, ones_matrix, zero_matrix)
        local_sim = torch.softmax(25 * cos_sim * thresh_map, dim=-1) * geo_sim
        local_sim = local_sim / torch.sum(local_sim, dim=-1, keepdim=True)
        local_sim = local_sim.to(features.dtype)
        sim_features = local_sim @ local_features

        return (sim_features + features) / 2

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, other_result={}, boxes=None, simN=30):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score)
        msa = self.linear1(trans_cls)
        # if other_result != []:
        #     other_msa = other_result['msa'].unsqueeze(0)
        #     msa = torch.cat([msa,other_msa],dim=1)
        ave = self.ave_pooling_over_ref(msa, sim_round2)
        out = self.linear2(ave)
        if other_result != [] and other_result['local_results'] != []:
            lout = self.local_agg(out[:simN], other_result['local_results'], boxes[:simN], cls_score[:simN],
                                  fg_score[:simN])
            return lout, out
        return out, out
