#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn


# from ..utils.feature_visualization import feature_visualization


class TSCD(nn.Module):
    """
    TSCD model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from TSCD head during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, time_embedding=None, nms_thresh=0.5, lframe=0, gframe=32, resume=False):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, \
                loss_matched_cls, \
                loss_matched_obj, \
                loss_matched_iou = self.head(
                    fpn_outs, targets, x, time_embedding, lframe=lframe, gframe=gframe, resume=resume)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "loss_matched_cls": loss_matched_cls,
                "loss_matched_obj": loss_matched_obj,
                "loss_matched_iou": loss_matched_iou
            }
        else:

            outputs = self.head(fpn_outs, targets, x, time_embedding, nms_thresh=nms_thresh, lframe=lframe, gframe=gframe,
                                resume=resume)

        return outputs