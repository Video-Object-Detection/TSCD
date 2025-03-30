#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

heatmap_path = "feature_visual/yolox_WaveletsHFBlock_start_0_36_feature_map_1.jpg"
img_path = "datasets/ILSVRC2015_val_00046000_panda/000491.JPEG"


def feature2heatmap(feature_map_path, img_path, save_path):
    """
        convert feature map to heatmap
    """
    feature_map = cv2.imread(feature_map_path)
    feature_map = np.array(feature_map)
    print(feature_map.shape)
    feature_map = np.maximum(feature_map, 0)
    feature_map = np.mean(feature_map, axis=2)
    feature_map /= np.max(feature_map)

    img = cv2.imread(img_path)
    print(img.shape)
    feature_map = cv2.resize(feature_map, (img.shape[1], img.shape[0]))
    feature_map = np.uint8(255 * feature_map)
    # Converting feature maps into different types of pseudo-colour maps
    heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    # combining feature maps with pseudo-colour maps
    heat_img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)
    cv2.imwrite(save_path, heat_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert the feature map to cam heatmap")
    parser.add_argument('--feature_map_path', type=str,
                        default='feature_visual/feature_map.jpg', help='feature map path')
    parser.add_argument('--image_path', type=str,
                        default='datasets/ILSVRC2025/ILSVRC2015_val_00046000_panda/000491.JPEG', help='image path')
    parser.add_argument('--heatmap_path', type=str, default='heatmap/cam.jpg', help='heatmap path')
    args = parser.parse_args()
