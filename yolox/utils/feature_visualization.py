#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
import numpy as np
import math


def feature_visualization(features, model_type, feature_num=4):
    """
        visual the feature map
        features: The feature map which you need to visualization
        feature_num: The amount of features to visualization you need
    """

    save_dir = "feature_visual/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(features, (list, tuple)):
        scale_num = 0
        for batch in features:
            batch_num = 0
            for feature in batch:
                feature.unsqueeze_(0)

                # block by channel dimension
                # blocks = torch.chunk(feature, feature.shape[1], dim=1)
                #
                # plt.figure()
                # for i in range(feature_num):
                #     torch.squeeze(blocks[i])
                #     feature = transforms.ToPILImage()(blocks[i].squeeze())
                #
                #     ax = plt.subplot(int(math.sqrt(feature_num)), int(math.sqrt(feature_num)), i+1)
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                #
                #     plt.imshow(feature)
                # plt.savefig(save_dir + 'yolox_{}_{}_{}_feature_map_{}.png'
                # .format(model_type, scale_num, batch_num, feature_num), dpi=300)

                blocks = feature[:, :3, :, :]
                scale = feature.shape[2]

                # print(blocks.shape)
                feature = transforms.ToPILImage()(blocks.squeeze())
                feature.save(
                    save_dir + 'yolox_{}_{}_{}_feature_map_{}.jpg'.format(model_type, batch_num, scale, feature_num))

                batch_num += 1
            scale_num += 1
    else:
        batch_num = 0
        for feature in features:
            feature.unsqueeze_(0)
            # block by channel dimension
            blocks = torch.mean(feature, dim=1, keepdim=True)
            blocks = blocks / torch.max(blocks)
            scale = feature.shape[2]
            # blocks = torch.chunk(feature, feature.shape[1], dim=1)
            #
            # plt.figure()
            # for i in range(feature_num):
            #     torch.squeeze(blocks[i])
            #     feature = transforms.ToPILImage()(blocks[i].squeeze())
            #
            #     ax = plt.subplot(int(math.sqrt(feature_num)), int(math.sqrt(feature_num)), i + 1)
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #
            #     plt.imshow(feature)
            # plt.savefig(save_dir + 'yolox_{}_{}_feature_map_{}.png'.format(model_type, batch_num, feature_num), dpi=300)
            feature = transforms.ToPILImage()(blocks.squeeze())
            feature.save(
                save_dir + 'yolox_{}_{}_{}_feature_map_{}.jpg'.format(model_type, batch_num, scale, feature_num))

            batch_num += 1


if __name__ == '__main__':
    # test
    features = torch.rand(16, 512, 16, 16)
    feature_visualization(features, 'backbone', 4)
