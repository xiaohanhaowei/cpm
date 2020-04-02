#!/home/wanghongwei/anaconda3/envs/tf114/bin/python
# coding=utf-8
# ===========================================
# ssd dataset enhance code
# ===========================================

import cv2
import cpm_utils
import numpy as np
import math
import tensorflow as tf
import time
import random
import os
import utils
import imagewarp
import copy

dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img/'
SHOW_INFO = False
box_size = 256
num_of_joints = 6
gaussian_radius = 2
img_count = 0
t1 = time.time()

gt_file = os.path.join(dataset_dir, 'labels.txt')
gt_content = open(gt_file, 'r').readlines()

for idx, line in enumerate(gt_content):
    line = line.split()

    # Check if it is a valid img file
    # import pdb
    # pdb.set_trace()
    if not line[0].endswith(('jpg', 'png')):
        continue
    cur_img_path = os.path.join(dataset_dir, line[0])
    print("starting to process", line[0])
    cur_img = cv2.imread(cur_img_path)

    # Read in bbox and joints coords
    tmp = [float(x) for x in line[1:5]]
    cur_hand_bbox = [min([tmp[0], tmp[2]]),
                     min([tmp[1], tmp[3]]),
                     max([tmp[0], tmp[2]]),
                     max([tmp[1], tmp[3]])
                     ]
    if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
    if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
    # FIXME: here may exist a bug
    # TODO: 各坐标对应的不同的轴在这困惑
    #       shape[0]:height，也就是竖着的方向有多大，竖着的方向为Y轴
    #       shape[1]:width，也就是横着方向有多大，横着的方向为X轴
    if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
    if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]
    # FIXME: need to modify the number of joints
    cur_hand_joints_x = [float(i) for i in line[7:17:2]]
    cur_hand_joints_x.append(float(line[5]))
    cur_hand_joints_y = [float(i) for i in line[8:17:2]]
    cur_hand_joints_y.append(float(line[6]))

    # Crop image and adjust joint coords
    cv2.namedWindow('orig')
    cv2.moveWindow('orig', 900, 0)
    cv2.imshow('orig', cur_img)
    cv2.waitKey(10)
    for degree in range(0, 360, 1):
        crop_img, joints_x, joints_y = imagewarp.crop_swarp(cur_img,
                                                            cur_hand_bbox, cur_hand_joints_x, cur_hand_joints_y, degree)
        crop_img_disp = copy.deepcopy(crop_img)
        # for i in range(len(cur_hand_joints_x)):
        #     cv2.circle(crop_img_disp, center=(int(joints_x[i]), int(joints_y[i])), radius=3, color=(0, 255, 255), thickness=-1)
        #     cv2.putText(crop_img_disp, str(i), (int(joints_x[i]), int(joints_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("crop", crop_img_disp)
        cv2.waitKey(1)

        output_image = np.ones(shape=(box_size, box_size, 3)) * 128
        output_heatmaps = np.zeros((box_size, box_size, num_of_joints))

        # Resize and pad image to fit output image size
        if crop_img.shape[0] > crop_img.shape[1]:
            scale = box_size / (crop_img.shape[0] * 1.0)

            # Relocalize points
            joints_x = map(lambda x: x * scale, joints_x)
            joints_y = map(lambda x: x * scale, joints_y)

            # Resize image 
            image = cv2.resize(crop_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            print('*' * 20)
            print('resized: ', image.shape)
            offset = image.shape[1] % 2

            output_image[:, int(box_size / 2 - math.floor(image.shape[1] / 2)): int(
                box_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
            joints_x = list(map(lambda x: x + (box_size / 2 - math.floor(image.shape[1] / 2)),
                                    joints_x))
            joints_y = list(joints_y)
            joints_x = np.asarray(joints_x)
            joints_y = np.asarray(joints_y)

            if SHOW_INFO:
                hmap = np.zeros((box_size, box_size))
                # Plot joints
                for i in range(num_of_joints):
                    cv2.circle(output_image, (int(joints_x[i]), int(joints_y[i])), 3, (0, 255, 0), 2)

                    # Generate joint gaussian map
                    part_heatmap = cpm_utils.gaussian_img(box_size, box_size, joints_x[i], joints_y[i], 1)
                    # part_heatmap = utils.make_gaussian(output_image.shape[0], gaussian_radius,
                     #                                  [joints_x[i], joints_y[i]])
                    hmap += part_heatmap * 50
            else:
                for i in range(num_of_joints):
                    # output_heatmaps[:, :, i] = utils.make_gaussian(box_size, gaussian_radius,
                    #                                               [joints_x[i], joints_y[i]])
                    output_heatmaps[:, :, i] = cpm_utils.gaussian_img(box_size, box_size, joints_x[i], joints_y[i], 1)

        else:
            scale = box_size / (crop_img.shape[1] * 1.0)

            # Relocalize points
            joints_x = list(map(lambda x: x * scale, joints_x))
            joints_y = list(map(lambda x: x * scale, joints_y))
            # Resize image
            image = cv2.resize(crop_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            offset = image.shape[0] % 2

            output_image[int(box_size / 2 - math.floor(image.shape[0] / 2)): int(
                box_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
            joints_y = list(map(lambda x: x + (box_size / 2 - math.floor(image.shape[0] / 2)),
                                    joints_y))

            joints_x = np.asarray(joints_x)
            joints_y = np.asarray(joints_y)

            if SHOW_INFO:
                hmap = np.zeros((box_size, box_size))
                # Plot joints
                for i in range(num_of_joints):
                    cv2.circle(output_image, (int(joints_x[i]), int(joints_y[i])), 3, (0, 255, 0), 2)

                    # Generate joint gaussian map
                    part_heatmap = utils.make_gaussian(output_image.shape[0], gaussian_radius,
                                                       [joints_x[i], joints_y[i]])
                    hmap += part_heatmap * 50
            else:
                for i in range(num_of_joints):
                    # FIXME: there is no function named `make_gaussian` in utils.py
                    output_heatmaps[:, :, i] = utils.make_gaussian(box_size, gaussian_radius,
                                                                   [joints_x[i], joints_y[i]])
        if SHOW_INFO:
            cv2.imshow('', hmap.astype(np.uint8))
            cv2.imshow('i', output_image.astype(np.uint8))
            cv2.waitKey(2)

        # Create background map
        output_background_map = np.ones((box_size, box_size)) - np.amax(output_heatmaps, axis=2)
        output_heatmaps = np.concatenate((output_heatmaps, output_background_map.reshape((box_size, box_size, 1))), axis=2)
        # cv2.imshow('', (output_background_map*255).astype(np.uint8))
        # cv2.imshow('h', (np.amax(output_heatmaps[:, :, 0:21], axis=2)*255).astype(np.uint8))
        # cv2.waitKey(1000)

        coords_set = np.concatenate((np.reshape(joints_x, (num_of_joints, 1)),
                                     np.reshape(joints_y, (num_of_joints, 1))),
                                    axis=1)

        output_image_raw = output_image.astype(np.uint8).tostring()
        output_heatmaps_raw = output_heatmaps.flatten().tolist()
        output_coords_raw = coords_set.flatten().tolist()

        # raw_sample = tf.train.Example(features=tf.train.Features(feature={
        #     'image': _bytes_feature(output_image_raw),
        #     'heatmaps': _float64_feature(output_heatmaps_raw)
        # }))

        img_count += 1
        if img_count % 50 == 0:
            print('Processed %d images, took %f seconds' % (img_count, time.time() - t1))
            t1 = time.time()
