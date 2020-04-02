#!/home/wanghongwei/anaconda3/envs/tf114/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def degree_compute(image, joints):
    base_vec = joints[0] - joints[5]
    l_edge_vec = joints[1] - joints[5]
    r_edge_vec = joints[2] - joints[5] 
    arrow_vec = joints[4] - joints[5]
    base_len = np.sqrt(base_vec[0]**2 + base_vec[1]**2)
    l_edge_len = np.sqrt(l_edge_vec[0]**2 + l_edge_vec[1]**2)
    r_edge_len = np.sqrt(r_edge_vec[0]**2 + r_edge_vec[1]**2)
    arrow_len = np.sqrt(arrow_vec[0]**2 + arrow_vec[1]**2)
    cos_theta0 = np.dot(base_vec, l_edge_vec) / (base_len * l_edge_len)
    cos_theta1 = np.dot(base_vec, r_edge_vec) / (base_len * r_edge_len)
    cos_theta2 = np.dot(base_vec, arrow_vec) / (base_len * arrow_len)
    cos_theta3 = np.dot(arrow_vec, l_edge_vec) / (arrow_len * l_edge_len)
    # cos_theta4 = np.dot(arrow_vec, r_edge_vec) / (arrow_len * r_edge_len)
    degree_flag = 5
    if cos_theta2 >= cos_theta0:  # theta2 <= theta0
        if cos_theta3 < cos_theta0:  # theta3 > theta0
            cv2.putText(image, 'high pressure', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            degree_flag = 4
        else:
            if cos_theta2 == cos_theta0:
                cv2.putText(image, 'low pressure attention', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                degree_flag = 1
            else:
                cv2.putText(image, 'low pressure', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                degree_flag = 0
    elif cos_theta2 < cos_theta0 and cos_theta2 >= cos_theta1:
        if cos_theta3 > cos_theta2:
            if cos_theta2 - cos_theta1 < 1e-1:
                cv2.putText(image, 'high pressure attention', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 255), 2)
                degree_flag = 3
            elif cos_theta0 - cos_theta2 < 1e-1:
                cv2.putText(image, 'low pressure attention', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                degree_flag = 1
            else:
                cv2.putText(image, 'OK', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                degree_flag = 2
        else:
            cv2.putText(image, 'high pressure', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            degree_flag = 4
    else:
        cv2.putText(image, 'high pressure', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        degree_flag = 4
    return degree_flag


def write_image(Full_Img, degree_flag, path_list):
    # cv2.imwrite('')
    pass
