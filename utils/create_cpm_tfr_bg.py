#!/home/wanghongwei/anaconda3/envs/tf114/bin/python
# env:python3
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
# for bg
tfr_file = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/tfrecords-from-cpm/bg/fas_bg_dataset.tfrecords'
dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/data_enhance/JPEGImages_source2'

SHOW_INFO = False
box_size = 256
num_of_joints = 6
gaussian_radius = 2


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create writer
tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0
t1 = time.time()
# added by hongwei.wang
# TODO: background image tfrecord generate
img_source = os.listdir(dataset_dir)
for s_img in img_source:
    if not s_img.endswith(('jpg', 'png')):
        continue
    cur_img_path = os.path.join(dataset_dir, s_img)
    output_image = cv2.imread(cur_img_path)
    # Crop image and adjust joint coords
    output_image = cv2.resize(output_image, (256, 256))
    cv2.namedWindow('orig')
    cv2.moveWindow('orig', 900, 0)
    cv2.imshow('orig', output_image)
    cv2.waitKey(10)
    output_image_raw = output_image.astype(np.uint8).tostring()
    output_image_shape_raw = np.array(output_image.shape).tolist()
    print(output_image_shape_raw)
    raw_sample = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(output_image_raw),
        'shape': _int64_feature(output_image_shape_raw)}))

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1
    if img_count % 50 == 0:
        print('Processed %d images, took %f seconds' % (img_count, time.time() - t1))
        t1 = time.time()
tfr_writer.close()
print('Done!')
