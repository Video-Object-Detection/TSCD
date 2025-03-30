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
        self.num_classes = 25
        self.seed = 2024
        # Define yourself dataset path
        self.data_dir = 'path to your datasets'
        self.train_ann = "ovis_train_new.json"
        self.val_ann = "ovis_valid_new.json"
        self.train_name = "train"
        self.val_name = "train"
        self.output_dir = "./OVIS_YOLOX_outputs/YOLOX_s"
        self.warmup_epochs = 1
        self.max_epoch = 7
        self.no_aug_epochs = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
