#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Occluded Video Instance Segmentation (OVIS) is used to validate the effect of detecting occluded objects.
The training set consists of 607 videos.
Because only the annotation of the training set is publicly available, the original OVIS training set
was partitioned into training and validation sets with an approximate ratio of 7:3 for each category.
"""
import argparse
import json
import random

import numpy as np
from utils.visual import draw_bar, draw_stacked_bar

# OVIS consists of 25 categories, where object occlusion usually occur.
Categories = ["Person", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe",
              "Poultry", "Giant_panda", "Lizard", "Parrot", "Monkey", "Rabbit", "Tiger", "Fish", "Turtle", "Bicycle",
              "Motorcycle", "Airplane", "Boat", "Vehical"]


def ovis_category_video_id(ovis_annotations):
    """
    Video ID corresponds to each category.
    :param ovis_annotations: OVIS annotations field
    """
    ovis_train_category_video_id = {}
    print("# Get the video id for each category")
    for anno in ovis_annotations:
        category_id = anno['category_id']
        video_id = anno['video_id']
        if category_id not in ovis_train_category_video_id:
            ovis_train_category_video_id[category_id] = [video_id]
        else:
            if video_id not in ovis_train_category_video_id[category_id]:
                ovis_train_category_video_id[category_id].append(video_id)
    json.dump(ovis_train_category_video_id, open('ovis_train_categroy_video_id.json', 'w+'), indent=1)
    print('# done \n')
    return ovis_train_category_video_id


def ovis_category_video_nums(ovis_train_category_video_id):
    category_video_nums = {}
    total = 0
    for i in range(len(ovis_train_category_video_id)):
        index = i + 1
        video_categroy = ovis_train_category_video_id[index]
        category_video_nums[index] = len(video_categroy)
        total += len(video_categroy)
    return category_video_nums, total


def ovis_division_for_categories(ovis_train_category_video_id, ratio=0.7):
    """
    Specified percentage of randomly selected videos from each category are used for training and validation.
    :param ovis_train_category_video_id: Video ID corresponds to each category.
    """
    print("# 70% of the videos from each category were randomly selected for training and 30% for testing")
    # video id for training and validation
    train_videos_id = []
    val_videos_id = []
    videos = {}
    random.seed(2024)

    for i in range(len(ovis_train_category_video_id)):
        index = i + 1
        video_num = len(ovis_train_category_video_id[index])
        num_video_to_select = round(video_num * ratio)
        # training video ids for each category
        train_videos_id_per = random.sample(ovis_train_category_video_id[index], num_video_to_select)
        # validation video ids for each category
        val_videos_id_per = []
        for id in ovis_train_category_video_id[index]:
            if id not in train_videos_id_per:
                val_videos_id_per.append(id)
        # add in training and validation sets
        for id in train_videos_id_per:
            if id not in train_videos_id and id not in val_videos_id:
                train_videos_id.append(id)
        for id in val_videos_id_per:
            if id not in train_videos_id and id not in val_videos_id:
                val_videos_id.append(id)

    print(f"train_videos:{len(train_videos_id)}")
    print(f"val_vidoes:{len(val_videos_id)}")
    train_videos_id = sorted(train_videos_id)
    val_videos_id = sorted(val_videos_id)
    videos["train"] = train_videos_id
    videos["val"] = val_videos_id
    json.dump(videos, open('ovis_train_division.json', 'w+'), indent=1)
    print("# done \n")

    return train_videos_id, val_videos_id


def division_train_valid_data(train_data_dir='annotations_train_original.json',
                              new_train_dir='annotations_train_modify1.json',
                              new_val_dir='annotations_val_modify1.json', ratio=0.7):
    """
    Divide the training and validation sets.
    :param train_data_dir: Path to the original training annotation file
    :param new_train_dir: Path to the new training annotation file
    :param new_val_dir: Path to the new validation annotation file
    """

    # OVIS VOD training and validation annotations
    ovis_vod_train = {}
    ovis_vod_val = {}

    # load the annotation of the original training sets
    with open(train_data_dir, 'r') as ovis:
        ovis = json.load(ovis)
    # copy related infos
    ovis_vod_train['info'] = ovis['info']
    ovis_vod_val['info'] = ovis['info']
    ovis_vod_train['categories'] = ovis['categories']
    ovis_vod_val['categories'] = ovis['categories']
    ovis_vod_train['licenses'] = ovis['licenses']
    ovis_vod_val['licenses'] = ovis['licenses']

    # get the video id for each category
    ovis_train_category_video_id = ovis_category_video_id(ovis["annotations"])

    # get the number of video for each category
    category_video_nums, total = ovis_category_video_nums(ovis_train_category_video_id)
    print(f"ovis_category_video_nums:{category_video_nums}")
    print(f"ovis_total_video_nums:{total}\n")

    # Specified percentage of randomly selected videos from each category are used for training and validation
    train_videos_id, val_videos_id = ovis_division_for_categories(ovis_train_category_video_id, ratio=ratio)

    # Populating the videos field in the training set and validation set annotations
    print("# filling videos for train and val")
    ovis_vod_train["videos"] = []
    ovis_vod_val["videos"] = []
    for id, video in enumerate(ovis["videos"]):
        id = id + 1
        if id in train_videos_id:
            index = train_videos_id.index(id)
            video["id"] = index + 1
            ovis_vod_train["videos"].append(video)
        else:
            index = val_videos_id.index(id)
            video["id"] = index + 1
            ovis_vod_val["videos"].append(video)
    t_video_nums = len(ovis_vod_train["videos"])
    v_video_nums = len(ovis_vod_val["videos"])
    print(f"OVID_VOD_train_videos_nums:{t_video_nums}")
    print(f"OVID_vod_val_videos_nums:{v_video_nums}")
    print("# done \n")

    # Populating the annotations field in the training set and validation set annotations
    print("# filling annotations for train and val")
    ovis_vod_train["annotations"] = []
    ovis_vod_val["annotations"] = []
    train_num = 0
    val_num = 0
    for anno in ovis["annotations"]:
        video_id = anno["video_id"]
        if video_id in train_videos_id:
            train_num += 1
            index = train_videos_id.index(video_id)
            anno["video_id"] = index + 1
            anno["id"] = train_num
            ovis_vod_train["annotations"].append(anno)
        else:
            val_num += 1
            index = val_videos_id.index(video_id)
            anno["video_id"] = index + 1
            anno["id"] = val_num
            ovis_vod_val["annotations"].append(anno)
    # print(train_num, val_num)
    t_annotation_nums = len(ovis_vod_train["annotations"])
    v_annotation_nums = len(ovis_vod_val["annotations"])
    print(f"OVIS_VOD_train_annotations_nums:{t_annotation_nums}")
    print(f"OVIS_VOD_val_annotations_nums:{v_annotation_nums}")
    print("# done \n")

    json.dump(ovis_vod_train, open(new_train_dir, 'w+'))
    json.dump(ovis_vod_val, open(new_val_dir, "w+"))

    # Get the number of different categories of targets and labels in the training set
    print("# get the number of objects and annotations for training")
    ovis_train_object_nums = {}
    ovis_train_annotations = {}
    for anno in ovis_vod_train["annotations"]:
        category_id = anno["category_id"]
        if category_id not in ovis_train_object_nums:
            ovis_train_object_nums[category_id] = 1
        else:
            ovis_train_object_nums[category_id] += 1
        for i in range(len(anno["bboxes"])):
            if anno['bboxes'][i] != None:
                if category_id not in ovis_train_annotations:
                    ovis_train_annotations[category_id] = 1
                else:
                    ovis_train_annotations[category_id] += 1
    print(f'OVIS_VOD_train_object_nums:{ovis_train_object_nums}')
    total_object_train = 0
    for index in ovis_train_object_nums:
        total_object_train += ovis_train_object_nums[index]
    print(f'OVIS_VOD_total_train_object_nums', total_object_train)
    print(f'OVIS_VOD_train_annotation_nums:{ovis_train_annotations}')
    total_annotation_train = 0
    for index in ovis_train_annotations:
        total_annotation_train += ovis_train_annotations[index]
    print(f'OVIS_VOD_total_train_annotation_nums:{total_annotation_train}')

    # Visualise the number of different categories of objects in the training set
    values_train_object_nums = [ovis_train_object_nums[i + 1] for i in range(len(ovis_train_object_nums))]
    # print(values_trian_object_nums)
    sorted_indices = np.argsort(-np.array(values_train_object_nums))
    sorted_values_train_object_nums = np.array(values_train_object_nums)[sorted_indices]
    sorted_categories_train_object_nums = np.array(Categories)[sorted_indices]
    draw_bar(sorted_categories_train_object_nums, sorted_values_train_object_nums,
             "The Number of Objects For Training Set", "Categories", "Number of Objects")

    # Visualise the number of different categories of annotations in the training set
    values_train_annotation_nums = [ovis_train_annotations[i + 1] for i in range(len(ovis_train_annotations))]
    # print(values_train_annotation_nums)
    sorted_indices = np.argsort(-np.array(values_train_annotation_nums))
    sorted_values_train_annotation_nums = np.array(values_train_annotation_nums)[sorted_indices]
    sorted_categories_train_annotation_nums = np.array(Categories)[sorted_indices]
    draw_bar(sorted_categories_train_annotation_nums, sorted_values_train_annotation_nums,
             "The Number of Labels For Training Set", "Categories", "Number of Labels")

    # Get the number of different categories of targets and labels in the validation set
    print("# get the number of objects and annotations for validation")
    ovis_val_object_nums = {}
    ovis_val_annotations = {}
    for anno in ovis_vod_val["annotations"]:
        category_id = anno["category_id"]
        if category_id not in ovis_val_object_nums:
            ovis_val_object_nums[category_id] = 1
        else:
            ovis_val_object_nums[category_id] += 1
        for i in range(len(anno["bboxes"])):
            if anno['bboxes'][i] != None:
                if category_id not in ovis_val_annotations:
                    ovis_val_annotations[category_id] = 1
                else:
                    ovis_val_annotations[category_id] += 1
    print(f'OVIS_VOD_valid_object_nums:{ovis_val_object_nums}')
    total_object_val = 0
    for index in ovis_val_object_nums:
        total_object_val += ovis_val_object_nums[index]
    print(f'OVIS_VOD_valid_total_object_nums:{total_object_val}')
    print(f'OVIS_VOD_valid_annotation_nums:{ovis_val_annotations}')
    total_annotation_val = 0
    for index in ovis_val_annotations:
        total_annotation_val += ovis_val_annotations[index]
    print(f'OVIS_VOD_valid_total_annotation_nums:{total_annotation_val}')

    # Visualise the number of different categories of objects in the validation set
    values_valid_object_nums = [ovis_val_object_nums[i + 1] for i in range(len(ovis_val_object_nums))]
    sorted_indices = np.argsort(-np.array(values_valid_object_nums))
    sorted_values_valid_object_nums = np.array(values_valid_object_nums)[sorted_indices]
    sorted_categories_valid_object_nums = np.array(Categories)[sorted_indices]
    draw_bar(sorted_categories_valid_object_nums, sorted_values_valid_object_nums,
             "The Number of Objects For Validation Set", "Categories", "Number of Objects")

    # Visualise the number of different categories of annotations in the validation set
    values_valid_annotation_nums = [ovis_val_annotations[i + 1] for i in range(len(ovis_val_annotations))]
    sorted_indices = np.argsort(-np.array(values_valid_annotation_nums))
    sorted_values_valid_annotation_nums = np.array(values_valid_annotation_nums)[sorted_indices]
    sorted_categories_valid_annotation_nums = np.array(Categories)[sorted_indices]
    draw_bar(sorted_categories_valid_annotation_nums, sorted_values_valid_annotation_nums,
             "The Number of Labels For Validation Set", "Categories", "Number of Labels")

    # Visualise the proportion of different categories of objects in the training and validation sets
    values_object = []
    values_object_val = []
    values_object_train = []
    for i in range(len(values_train_object_nums)):
        values_object.append(values_valid_object_nums[i] + values_train_object_nums[i])
    for i in range(len(values_object)):
        values_object_val.append(values_valid_object_nums[i] / values_object[i])
    for i in range(len(values_object)):
        values_object_train.append(values_train_object_nums[i] / values_object[i])
    draw_stacked_bar(Categories, values_object_val, values_object_train,
                     "Object Ratios by Category in Dataset", "Categories", "Object Ratios")

    # Visualise the proportion of different categories of annotations in the training and validation sets
    values_annotation = []
    values_annotation_val = []
    values_annotation_train = []
    for i in range(len(values_train_annotation_nums)):
        values_annotation.append(values_valid_annotation_nums[i] + values_train_annotation_nums[i])
    for i in range(len(values_annotation)):
        values_annotation_val.append(values_valid_annotation_nums[i] / values_annotation[i])
    for i in range(len(values_annotation)):
        values_annotation_train.append(values_train_annotation_nums[i] / values_annotation[i])
    draw_stacked_bar(Categories, values_annotation_val, values_annotation_train,
                     "Label Ratios by Category in Dataset", "Categories", "Label Ratios")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OVIS_Division')
    parser.add_argument("-o", "--original_dir", type=str,
                        default="annotations_train_original.json", help="original training annotations")
    parser.add_argument("-t", "--new_train_dir", type=str,
                        default="annotations_train_new.json", help="new training annotations")
    parser.add_argument("-v", "--new_val_dir", type=str,
                        default="annotations_valid_new.json", help="new validation annotations")
    parser.add_argument("-ratio", "--ratio", type=float, default=0.7, help="ratio of new training data")
    args = parser.parse_args()
    # Divide the training and validation sets and visualise the ratios
    division_train_valid_data(args.original_dir, args.new_train_dir, args.new_val_dir, args.ratio)
