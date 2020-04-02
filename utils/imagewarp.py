import os
import cv2
import numpy as np


#dataset_dir = '/home/lynxi/CPM/img' # 原始图片集
#gt_file = os.path.join(dataset_dir, 'labels.txt') # label
#gt_content = open(gt_file, 'r').readlines()


def swarp(image, srcPoints, rotate_center, angle):
    dstPoints = list()
    h, w, _ = image.shape

    for point in srcPoints:
        x1 = point[0]
        y1 = h - point[1]

        x2 = rotate_center[0]
        y2 = h - rotate_center[1]

        x = np.round((x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2)
        y = np.round((x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2)
        y = h - y
        dstPoints.append([x, y])

    return dstPoints


def crop_swarp(oriimg, cur_hand_bbox, cur_hand_joints_x, cur_hand_joints_y, angle=-90):

    oriPoints = [[x, y] for x, y in zip(cur_hand_joints_x, cur_hand_joints_y)]

    # 以最后一个点为旋转中心
    Center = oriPoints[-1]
    H, W = oriimg.shape[:2]
    M = cv2.getRotationMatrix2D((Center[0], Center[1]), angle, 1)
    warpimg = cv2.warpAffine(oriimg, M, (W, H), borderValue=(128, 128, 128))

    warpPoints = swarp(oriimg, oriPoints, oriPoints[-1], angle)  # -90° 顺时针旋转 +90°逆时针旋转

    crop = warpimg[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),
           int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])), :]

    joints_x = [P[0] - cur_hand_bbox[0] for P in warpPoints]
    joints_y = [P[1] - cur_hand_bbox[1] for P in warpPoints]

    return crop, joints_x, joints_y
'''
for idx, line in enumerate(gt_content):
    line = line.split()

    if not line[0].endswith(('jpg', 'png')):
        continue
    cur_img_path = os.path.join(dataset_dir, line[0])
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

    crop, joints_x, joints_y = crop_swarp(cur_img, cur_hand_joints_x, cur_hand_joints_y, -90) #顺时针旋转90°

    for i in range(len(cur_hand_joints_x)):
        cv2.circle(crop, center=(int(joints_x[i]), int(joints_y[i])), radius=3, color=(0, 255, 255), thickness=-1)
        cv2.putText(crop, str(i), (int(joints_x[i]), int(joints_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("crop", crop)

    ###############
    cv2.waitKey(0)
'''
