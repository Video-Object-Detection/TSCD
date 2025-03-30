#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        # self.data_dir = '/mnt/weka/scratch/datasets/coco/'
        self.num_classes = 80
        self.seed = 2024
        self.data_dir = 'path to your datasets'
        self.train_ann = "annotations/ILSVRC_FGFA_COCO.json"
        self.val_ann = "annotations/vid_val10000_coco.json"
        self.output_dir = "./YOLOX_outputs"
        self.warmup_epochs = 1
        self.max_epoch = 7
        self.no_aug_epochs = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
