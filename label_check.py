# coding=utf-8
import cv2
import os
import copy as cp
# `base` the path that contain the images
# `label` the label file's name
# Need modify both the `base` and `label` var!!!
# Need move the 'label' var ie. the joints' file under the `base` dir
base = ''
label = ''
# eg:
# base = './cap0/img'
# label = 'img.txt'
f = open(os.path.join(base, label), 'r')
lines = f.readlines()
for line in lines:
    line = line.split()
#     assert len(line)==5
    img_name = line[0]
    print(img_name)
    warpedimg = cv2.imread(os.path.join(base, img_name))
    warpedimg_disp = cp.deepcopy(warpedimg)
    print(warpedimg_disp.shape)
    bbox_lx = int(line[1])
    bbox_ly = int(line[2])
    bbox_rx = int(line[3])
    bbox_ry = int(line[4]) 
    cv2.circle(warpedimg_disp, (bbox_lx, bbox_ly), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.putText(warpedimg_disp, 'left', (bbox_lx, bbox_ly), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
    cv2.circle(warpedimg_disp, (bbox_rx, bbox_ry), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.putText(warpedimg_disp, 'right', (bbox_rx, bbox_ry), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
    cv2.rectangle(warpedimg_disp, (bbox_lx, bbox_ly), (bbox_rx, bbox_ry), (255, 0, 0), 1)
    for i in range(5, 17, 2):
        cv2.circle(warpedimg_disp, (int(line[i]), int(line[i + 1])), 1, (0, 255, 0), 2)
        cv2.putText(warpedimg_disp, str(int((i - 5) / 2)), (int(line[i]) + 3, int(line[i + 1]) + 3), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
    cv2.namedWindow('img-orig')
    cv2.moveWindow('img', 600, 0)
    cv2.imshow('img-orig', warpedimg)
    cv2.namedWindow('img')
    cv2.moveWindow('img', 1000, 0)
    cv2.imshow('img', warpedimg_disp)
#     cv2.imwrite(os.path.join(base, 'ex.jpg'), warpedimg_disp)
    c = cv2.waitKey(0)
    if c == ord('q'):
        cv2.destroyAllWindows()
        break
