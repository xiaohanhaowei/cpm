#!/home/wanghongwei/anaconda3/envs/tf114/bin/python
# For single hand and no body part in the picture
# ======================================================

import math
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import sys
sys.path.append('../')
from config import FLAGS
from models.nets import cpm_hand
from degree_compute import degree_compute


def model_save():
    with tf.Session() as sess:
        # #(2) 第二种模型加载方法
        # saver = tf.train.import_meta_graph('./models/weights/0102/init_0.001_rate_0.5_step_10000-33000.meta')
        # saver.restore(sess, "./models/weights/0102/init_0.001_rate_0.5_step_10000-33000")

        # #(1) 第一种模型加载方法,输入节点重命名
        with gfile.FastGFile('./models/weights/cpmsave/fascpm_freeze0102.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input0 = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name="input")
            tf.import_graph_def(graph_def, input_map={"input_placeholder:0": input0},
                                return_elements=['stage_3/mid_conv7/BiasAdd'], name='')  # 导入计算图

        # #(１)第一种pb 模型保存
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['stage_3/mid_conv7/BiasAdd'])
        with tf.gfile.FastGFile('./models/weights/cpmsave/fascpm_freeze0102_new.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())


class FasCPM(object):
    boxsize = 256
    heatmap_size = 32
    num_joints = 6
    degree_en = False

    def __init__(self, model_path, boxsize=256, heatmap_size=32, num_joints=6):
        self.model_path = model_path
        self.boxsize = boxsize
        self.heatmap_size = heatmap_size
        self.num_joints = num_joints

        # ## (1)
        """
        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_graph = tf.get_default_graph()
        tf_graph.device('/device:GPU:0')
        tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=tf_graph, config=tf_config)
        self.input = "input_placeholder:0"
        self.output = tf.get_default_graph().get_tensor_by_name('stage_3/mid_conv7/BiasAdd:0')
        """

        # # (2)
        saver = tf.train.import_meta_graph(self.model_path + ".meta")
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_graph = tf.get_default_graph()
        tf_graph.device('/device:GPU:0')
        self.sess = tf.Session(graph=tf_graph, config=tf_config)
        saver.restore(self.sess, self.model_path)
        self.input = "input_placeholder:0"
        self.output = tf.get_default_graph().get_tensor_by_name('stage_3/mid_conv7/BiasAdd:0')

        # # (3)
        # ckpt = tf.train.get_checkpoint_state(self.model_path)
        # # saver = tf.train.import_meta_graph(self.model_path+".meta")
        # saver = tf.train.Saver()
        # tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        # tf_graph = tf.get_default_graph()
        # tf_graph.device('/device:GPU:0')
        # self.sess = tf.Session(graph=tf_graph, config=tf_config)
        # # saver.restore(self.sess, self.model_path)
        # saver.restore(self.sess, ckpt.model_checkpoint_path)
        # self.input = "input_placeholder:0"
        # self.output = tf.get_default_graph().get_tensor_by_name('stage_3/mid_conv7/BiasAdd:0')

        # ##(4)
        # self.model = cpm_hand.CPM_Model(input_size=FLAGS.input_size,
        #                         heatmap_size=FLAGS.heatmap_size,
        #                         stages=FLAGS.cpm_stages,
        #                         joints=FLAGS.num_of_joints,
        #                         img_type=FLAGS.color_channel,
        #                         is_training=False)
        # _ = tf.Variable(initial_value='fake_variable')
        # saver = tf.train.Saver()
        # self.output = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)
        # device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
        # sess_config = tf.ConfigProto(device_count=device_count)
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        # sess_config.gpu_options.allow_growth = True
        # sess_config.allow_soft_placement = True
        # self.sess = tf.Session(config=sess_config)
        # # for key in FLAGS.flag_values_dict():
        # #     print(key, FLAGS[key].value, type(FLAGS[key].value))
        # print(FLAGS.pretrained_model)
        # # pre_tr_model_path = str(FLAGS.model_path)
        # ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model)
        # print('*' * 20, FLAGS.pretrained_model, ckpt)  # , os.system('ls %s') % pre_tr_model_path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     print("m" * 20, '\n', "load from ckpt")
        #     print(ckpt.model_checkpoint_path)
        #     saver.restore(self.sess, ckpt.model_checkpoint_path)
        # for variable in tf.trainable_variables():
        #     with tf.variable_scope('', reuse=True):
        #         var = tf.get_variable(variable.name.split(':0')[0])
        #         print(variable.name, np.mean(sess.run(var)))
        # print('*#' * 20)
        # import pdb
        # pdb.set_trace()

    @staticmethod
    def circle(image):

        gimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(gimage, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=100, minRadius=20, maxRadius=500)
        if circles is None:
            print("No circles !")
            return None, None, None, None, None

        # 根据半径，找最大的圆
        Id = np.argmax(circles[:, :, 2])
        x, y, r = circles[0, Id]
        left = int(max(x - 1.2 * r, 0))
        top = int(max(y - 1.2 * r, 0))
        right = int(min(x + 1.2 * r, image.shape[1]))
        bot = int(min(y + 1.2 * r, image.shape[0]))
        Roimg = image[top:bot, left:right, :]

        # # x, y, R
        # for cir in circles[0, :]:
        #     x, y, r = cir
        #     cv2.circle(image, (x, y), r, (0, 0, 255), 2)
        # cv2.imshow("image", image)

        return Roimg, left, top, right, bot

    def img_white_balance(self, img, white_ratio):
        for channel in range(img.shape[2]):
            channel_max = np.percentile(img[:, :, channel], 100 - white_ratio)
            channel_min = np.percentile(img[:, :, channel], white_ratio)
            img[:, :, channel] = (channel_max - channel_min) * (img[:, :, channel] / 255.0)
        return img

    @classmethod
    def preprocesss(cls, image):

        img_h, img_w, _ = image.shape

        if img_h > img_w:
            scale = cls.boxsize / img_h
        else:
            scale = cls.boxsize / img_w

        resize = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w, _ = resize.shape
        # h, w <= box_size

        # output_img = np.ones((self.boxsize, self.boxsize, 3)) * 128
        # output_img[int(np.floor((self.boxsize - h)/2)):int(np.ceil((self.boxsize + h)/2)),
        # int(np.floor((self.boxsize - w) / 2)):int(np.ceil((self.boxsize + w) / 2)), :] = resize

        pad_h_offset = (cls.boxsize - h) % 2
        pad_w_offset = (cls.boxsize - w) % 2
        pad_boundary = [(cls.boxsize - h) // 2 + pad_h_offset, (cls.boxsize - h) // 2,
                        (cls.boxsize - w) // 2 + pad_w_offset, (cls.boxsize - w) // 2]

        output_img = cv2.copyMakeBorder(resize,
                                        top=pad_boundary[0],
                                        bottom=pad_boundary[1],
                                        left=pad_boundary[2],
                                        right=pad_boundary[3],
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=(128, 128, 128))
        # output_img = self.img_white_balance(output_img, 5)

        # output_img = np.ones((self.boxsize, self.boxsize, 3)) * 128
        # offset = h % 2
        # output_img[int(self.boxsize / 2 - math.floor(h / 2)): int(self.boxsize / 2 + math.floor(h / 2) + offset),
        #             int(self.boxsize / 2 - math.floor(w / 2)): int(self.boxsize / 2 + math.floor(w / 2) + offset), :] = resize

        output_img = output_img / 255.0 - 0.5

        return output_img

    # @classmethod
    def forward(self, input):
        input = np.expand_dims(input, axis=0)  # 扩一维
        outputs = self.sess.run(self.output, feed_dict={self.input: input})
        return outputs

    @classmethod
    def draw_joints(cls, full_img, joint_coords):

        # Plot joints
        for joint_num in range(cls.num_joints):
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3, color=(255, 0, 0), thickness=-1)
            cv2.putText(full_img, str(joint_num), (int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # # Plot limbs 连接关键点
        # for limb_num in range(len(self.limbs)):
        #     x1 = int(joint_coords[int(self.limbs[limb_num][0])][0])
        #     y1 = int(joint_coords[int(self.limbs[limb_num][0])][1])
        #     x2 = int(joint_coords[int(self.limbs[limb_num][1])][0])
        #     y2 = int(joint_coords[int(self.limbs[limb_num][1])][1])
        #     length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        #     if length < 150 and length > 5:
        #         deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        #         polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
        #                                    (int(length / 2), 3),
        #                                    int(deg),
        #                                    0, 360, 1)
        #         cv2.fillConvexPoly(full_img, polygon, color=(255,0,0))

    @classmethod
    def draw_joints_2(cls, full_img, l, t, r, b, joint_coords):

        # Plot joints
        cv2.rectangle(full_img, (l, t), (r, b), (255, 0, 0), 3)
        for joint_num in range(cls.num_joints):
            cv2.circle(full_img, center=(int(joint_coords[joint_num][0]), int(joint_coords[joint_num][1])), radius=3, color=(255, 0, 0), thickness=-1)
            cv2.putText(full_img, str(joint_num), (int(joint_coords[joint_num][0]), int(joint_coords[joint_num][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    @classmethod
    def correct_and_draw(cls, FullImg, image, last_heatmaps, crop_img, degree_en):
        '''
        param:
            image --> RoiImg
        return:
            local_joint_coord_set
            joint_coord_set [num_joints, 2] RoiImg's joints
        '''
        joint_coord_set = np.zeros((cls.num_joints, 2))
        local_joint_coord_set = np.zeros((cls.num_joints, 2))

        img_h, img_w, _ = image.shape
        scale = cls.boxsize / max(img_h, img_w)

        new_h = img_h * scale
        new_w = img_w * scale

        # Plot joint colors
        for joint_num in range(cls.num_joints):
            tmp_heatmap = last_heatmaps[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap), (cls.boxsize, cls.boxsize))
            joint_coord = np.array(joint_coord).astype(np.float32)
            local_joint_coord_set[joint_num, :] = joint_coord

            # Substract padding border 返回到原图
            joint_coord[0] = (joint_coord[0] - (cls.boxsize - new_h) / 2.) / scale  # y
            joint_coord[1] = (joint_coord[1] - (cls.boxsize - new_w) / 2.) / scale  # x
            print("joint_coord", joint_coord)

            joint_coord_set[joint_num, :] = joint_coord
        cls.draw_joints(image, joint_coord_set)
        if degree_en:
            degree_flag = degree_compute(FullImg, joint_coord_set)
            return joint_coord_set, local_joint_coord_set, degree_flag
        else:
            return joint_coord_set, local_joint_coord_set, None
        # cls.draw_joints(crop_img, local_joint_coord_set)

    def draw_heatmap(self, FullImg, RoiImg, heatmaps, crop_img, degree_en):
        print("heatmaps", len(heatmaps[0, :, :, 0:self.num_joints]), heatmaps[0, :, :, 0:self.num_joints].shape)
        last_heatmap = heatmaps[0, :, :, 0:self.num_joints].reshape(
            (self.heatmap_size, self.heatmap_size, self.num_joints))
        last_heatmap = cv2.resize(last_heatmap, (self.boxsize, self.boxsize))
        return self.correct_and_draw(FullImg, RoiImg, last_heatmap, crop_img, degree_en)

    def draw_new(self, stage_heatmap_np):

        demo_stage_heatmap = stage_heatmap_np[0, :, :, 0:self.num_joints].reshape(
            (self.heatmap_size, self.heatmap_size, self.num_joints))
        demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (self.boxsize, self.boxsize))
        demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
        demo_stage_heatmap = np.reshape(demo_stage_heatmap, (self.boxsize, self.boxsize, 1))
        demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
        cv2.namedWindow("current heatmap", 0)
        cv2.imshow("current heatmap", (demo_stage_heatmap * 255).astype(np.uint8))

    # @staticmethod
    def detect(self, test_path, img_name, image):

        RoiImg, left, top, right, bottom = self.circle(image)

        cropimg = None
        if RoiImg is None:
            print("No circle Roi!")
            return image, RoiImg, cropimg, None
        cv2.imwrite(os.path.join(test_path, '{}_circled.jpg'.format(img_name.split('.')[0])), RoiImg)
        cropimg = self.preprocesss(RoiImg)

        outputs = self.forward(cropimg)
        print("outputs.shape", outputs.shape)

        _, _, degree_flag = self.draw_heatmap(image, RoiImg, outputs, cropimg, self.degree_en)
        self.draw_new(outputs)

        return image, RoiImg, cropimg, degree_flag

    # @staticmethod
    def perform_eval(self, image, left, top, right, bottom, gt_joints):
        '''
        gt_joints: The ground-truth joints in consistent with the prediction
        of model predicting
        '''

        # RoiImg, left, top, right, bottom = self.circle(image)
        RoiImg = image[int(top):int(bottom), int(left):int(right)]
        cropimg = None
        if RoiImg is None:
            print("No circle Roi!")
            return image, RoiImg, cropimg

        cropimg = self.preprocesss(RoiImg)

        outputs = self.forward(cropimg)
        print("outputs.shape", outputs.shape)
        # TODO:add the cosin-sim method
        joint_coord_set, _ = self.draw_heatmap(RoiImg, outputs, cropimg)
        if gt_joints is not None:
            dist = self._distance_cosine(joint_coord_set.flatten(), np.array(gt_joints).flatten())
        else:
            dist = None
        self.draw_new(outputs)

        return image, RoiImg, cropimg, joint_coord_set.flatten(), dist

    @staticmethod
    def _distance_cosine(vector1, vector2):

        dot_res = np.dot(vector1, vector2)
        mul_res = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos = dot_res / mul_res
        sim = cos
        return sim


#  Cam or Video
def main_video():
    # init
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.6_step_10000-100000"
    # model_filename = "/home/wanghongwei/WorkSpace/detect/cpm-tf/models/weights/cpm_hand/finetune-0115/init_0.075_rate_0.73_step_10000-190000"
    # model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    # model_path = FLAGS.pretrained_model
    fascpm = FasCPM(FLAGS.pretrained_model_file)
    detes = list()
    frame_num = 0
    Write_En = False
    result_base_path = ['/home/wanghongwei/WorkSpace/datasets/fans/datasets/result' for num in range(5)]
    arrow_state = ['low', 'low_a', 'ok', 'h_a', 'h']
    result_path_l = list(map(os.path.join, result_base_path, arrow_state))
    if not os.path.exists(result_base_path[0]):
        os.mkdir(result_base_path[0])
        list(map(os.mkdir, result_path_l))
    elif not os.path.exists(result_path_l[0]):
        list(map(os.mkdir, result_path_l))

    # cap = cv2.VideoCapture("/home/lynxi/darknet/a.mp4")

    cap = cv2.VideoCapture(0)
    isOpened = cap.isOpened()
    if isOpened is False:
        print("No Cam video !")
        exit(1)
    while (isOpened):
        ret, frame = cap.read()
        if ret is False:
            print("No frame !")
            exit(1)
        frame_num += ret
        frame, dete, crop, degree_flag = fascpm.detect(frame)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", frame)
        if Write_En:
            if degree_flag not in range(5):
                continue
            else:
                print(degree_flag)
                print(frame_num)
                cv2.imwrite(os.path.join(result_path_l[degree_flag], (str(frame_num) + '.jpg')), frame)
        if dete is None:
            print("No dete !")
            if len(detes) > 1:
                dete = detes[-1]
            else:
                continue
        else:
            detes.append(dete)
            if len(detes) > 3:
                detes.pop(0)
        try:
            cv2.namedWindow('dete', 0)
            cv2.imshow('dete', dete)
            cv2.namedWindow("crop", 0)
            cv2.imshow("crop", crop)
        except:
            continue

        if cv2.waitKey(1) == 27:
            break
        # elif frame_num == 1000000:
        #     break
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()


def main_imgs_no_label():
    # init
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.6_step_10000-100000"
    # model_filename = "/home/wanghongwei/WorkSpace/detect/cpm-tf/models/weights/cpm_hand/finetune-0115/init_0.075_rate_0.73_step_10000-190000"
    # model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    # model_path = FLAGS.pretrained_model
    fascpm = FasCPM(FLAGS.pretrained_model_file)
    detes = list()
    frame_num = 0
    Write_En = False
    if Write_En:
        result_base_path = ['/home/wanghongwei/WorkSpace/datasets/fans/datasets/result' for num in range(5)]
        arrow_state = ['low', 'low_a', 'ok', 'h_a', 'h']
        result_path_l = list(map(os.path.join, result_base_path, arrow_state))
        if not os.path.exists(result_base_path[0]):
            os.mkdir(result_base_path[0])
            list(map(os.mkdir, result_path_l))
        elif not os.path.exists(result_path_l[0]):
            list(map(os.mkdir, result_path_l))

    # cap = cv2.VideoCapture("/home/lynxi/darknet/a.mp4")
    test_path = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img_test'
    img_list = os.listdir(test_path)
    for img_name in img_list:
        # s_img_name = os.path.join(img_dir, img_name)
        frame = cv2.imread(os.path.join(test_path, img_name))
        frame, dete, crop, degree_flag = fascpm.detect(test_path, img_name, frame)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", frame)
        cv2.imwrite(os.path.join(test_path, '{}_deted.jpg'.format(img_name.split('.')[0])), frame)
        if Write_En:
            if degree_flag not in range(5):
                continue
            else:
                print(degree_flag)
                print(frame_num)
                cv2.imwrite(os.path.join(result_path_l[degree_flag], (str(frame_num) + '.jpg')), frame)
        if dete is None:
            print("No dete !")
            if len(detes) > 1:
                dete = detes[-1]
            else:
                continue
        else:
            detes.append(dete)
            if len(detes) > 3:
                detes.pop(0)
        try:
            cv2.namedWindow('dete', 0)
            cv2.imshow('dete', dete)
            cv2.namedWindow("crop", 0)
            cv2.imshow("crop", crop)
        except:
            continue

        if cv2.waitKey(0) == 27:
            break
        # elif frame_num == 1000000:
        #     break
        else:
            continue

    cv2.destroyAllWindows()


# some imgs
def main_imgs():
    # init
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.6_step_10000-55000"
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.5_step_10000-270000"

    # model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    fascpm = FasCPM(FLAGS.pretrained_model_file)
    detes = list()

    dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img'  # 原始图片集
    gt_file = os.path.join(dataset_dir, 'labels.txt')  # label
    gt_content = open(gt_file, 'r').readlines()

    for idx, line in enumerate(gt_content):
        line = line.split()
        if not line[0].endswith(('jpg', 'png')):
            continue
        cur_img_path = os.path.join(dataset_dir, line[0])
        cur_img = cv2.imread(cur_img_path)
        print(line[0])
        cv2.destroyAllWindows()
        cv2.namedWindow("image-{}".format(line[0]), 0)
        cv2.moveWindow("image-{}".format(line[0]), 1500, 0)
        cv2.imshow("image-{}".format(line[0]), cur_img)
        cv2.waitKey(0)
        image, dete, crop = fascpm.detect(cur_img)
        cv2.destroyAllWindows()
        cv2.namedWindow("image-{}".format(line[0]), 0)
        cv2.moveWindow("image-{}".format(line[0]), 1500, 0)
        cv2.imshow("image-{}".format(line[0]), image)

        if dete is None:
            print("Can not find circle")
            if len(detes) > 1:
                dete = detes[-1]
            else:
                continue
        else:
            detes.append(dete)
            if len(detes) > 3:
                detes.pop(0)

        # cv2.namedWindow("dete", 0)
        # cv2.imshow("dete", dete)

        # cv2.namedWindow("crop", 0)
        # cv2.imshow("crop", crop + 0.5)

        if cv2.waitKey(10) == ord('q'):
            cv2.destroyAllWindows()
            break


# evaluate test-imgs
def eval_imgs():
    # init
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.6_step_10000-55000"
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.5_step_10000-270000"

    # model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    fascpm = FasCPM(FLAGS.pretrained_model_file)
    detes = list()

    # dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img'  # 原始图片集
    dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/Eval'  # 原始图片集
    gt_file = os.path.join(dataset_dir, 'labels.txt')  # label
    res_file = os.path.join(dataset_dir, 'result.txt')
    gt_content = open(gt_file, 'r').readlines()
    # if not os.path.isfile(res_file):
    result = open(res_file, 'a+')

    for idx, line in enumerate(gt_content):
        line = line.split()
        if not line[0].endswith(('jpg', 'png')):
            continue
        cur_img_path = os.path.join(dataset_dir, line[0])
        cur_img = cv2.imread(cur_img_path)
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
        # FIXME: need to transfer to the ground-truth joints
        cur_hand_joints_x = [float(i) for i in line[7:17:2]]
        cur_hand_joints_x.append(float(line[5]))
        cur_hand_joints_y = [float(i) for i in line[8:17:2]]
        cur_hand_joints_y.append(float(line[6]))
        gt_joints = [[cur_hand_joints_x[i], cur_hand_joints_y[i]] for i in range(FLAGS.num_of_joints)]
        # the former statement may be false
        image, dete, crop, _, dist = fascpm.perform_eval(cur_img, cur_hand_bbox[0], cur_hand_bbox[1], cur_hand_bbox[2], cur_hand_bbox[3], gt_joints)
        print(dist)
        to_write_string = line[0] + ' ' + str(dist) + '\n'
        result.write(to_write_string)
        # import pdb
        # pdb.set_trace()
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)

        if dete is None:
            print("Can not find circle")
            if len(detes) > 1:
                dete = detes[-1]
            else:
                continue
        else:
            detes.append(dete)
            if len(detes) > 3:
                detes.pop(0)

        # cv2.namedWindow("dete", 0)
        # cv2.imshow("dete", dete)

        cv2.namedWindow("crop", 0)
        cv2.imshow("crop", crop + 0.5)

        cv2.waitKey(10)
    result.close()


def eval_SSD():
    # init
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.6_step_10000-55000"
    # model_filename = "../models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000/init_0.01_rate_0.5_step_10000-270000"

    # model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    fascpm = FasCPM(FLAGS.pretrained_model_file)
    detes = list()

    # dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img'  # 原始图片集
    dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img'  # 原始图片集
    gt_file = os.path.join(dataset_dir, 'labels.txt')  # label
    SSD_output_file = os.path.join(dataset_dir, 'ssd-detect-newData.txt')
    new_labels_file = os.path.join(dataset_dir, 'labels_by_ssd.txt')
    cmp_file = os.path.join(dataset_dir, 'result_by_SSD.txt')
    gt_content = open(SSD_output_file, 'r').readlines()
    result = open(new_labels_file, 'a+')

    for idx, line in enumerate(gt_content):
        line = line.split()
        if not line[0].endswith(('jpg', 'png')):
            continue
        cur_img_path = os.path.join(dataset_dir, line[0])
        cur_img = cv2.imread(cur_img_path)
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
        # FIXME: need to transfer to the ground-truth joints
        # cur_hand_joints_x = [float(i) for i in line[7:17:2]]
        # cur_hand_joints_x.append(float(line[5]))
        # cur_hand_joints_y = [float(i) for i in line[8:17:2]]
        # cur_hand_joints_y.append(float(line[6]))
        # gt_joints = [[cur_hand_joints_x[i], cur_hand_joints_y[i]] for i in range(FLAGS.num_of_joints)]
        gt_joints = None
        # the former statement may be false
        image, dete, crop, joints_inf, dist = fascpm.perform_eval(cur_img, cur_hand_bbox[0], cur_hand_bbox[1], cur_hand_bbox[2], cur_hand_bbox[3], gt_joints)
        # print(dist)
        to_write_string = ''
        for i in range(5):
            to_write_string += line[i] + ' '
        joints_inf_len = len(joints_inf)
        for i in range(joints_inf_len):
            to_write_string += str(int(joints_inf[i])) + ' '
        to_write_string += '\n'
        result.write(to_write_string)
        # import pdb
        # pdb.set_trace()
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)

        if dete is None:
            print("Can not find circle")
            if len(detes) > 1:
                dete = detes[-1]
            else:
                continue
        else:
            detes.append(dete)
            if len(detes) > 3:
                detes.pop(0)

        # cv2.namedWindow("dete", 0)
        # cv2.imshow("dete", dete)

        cv2.namedWindow("crop", 0)
        cv2.imshow("crop", crop + 0.5)

        cv2.waitKey(10)
    result.close()
    return gt_file, new_labels_file, cmp_file


#  only one image
def main_img():
    # model_filename = "./models/weights/0107/init_0.001_rate_0.5_step_10000-100000"
    model_filename = "./models/weights/cpmsave/fascpm_freeze0102.pb"
    fascpm = FasCPM(model_filename)
    img = cv2.imread("./test_imgs/138.jpg")

    image, dete, crop = fascpm.detect(img)

    cv2.namedWindow("image", 0)
    cv2.imshow("image", image)

    if dete is None:
        print("Can not find circle")
    else:
        cv2.namedWindow("dete", 0)
        cv2.imshow("dete", dete)
    cv2.waitKey(1)


def cmp_result(labels_ref, labels_cmp, cmp_file):
    '''
        labels_ref refers to the reference lables
        labels_cmp refers to the lables file that generate from SSD model
    '''
    cmp_res = open(cmp_file, 'a')
    ref_file = open(labels_ref, 'r')
    cmp_file = open(labels_cmp, 'r')
    ref_content = ref_file.readlines()
    cmp_content = cmp_file.readlines()
    # cmp_content_trans = []
    # for line in cmp_content:
    #     s = [x for x in line.strip().split()]
    #     cmp_content_trans.append(s)
    # cmp_content_trans = sorted(cmp_content_trans, key=lambda label: int(label[0].split('.')[0]))
    ref_len = len(ref_content)
    cmp_len = len(cmp_content)
    print(ref_len)
    print(cmp_len)
    # if ref_len != cmp_len:
    #     raise ValueError('reference file\'s size is not accordance with the compare file')
    # else:
    for idx, ref_line in enumerate(ref_content):
        # ref_line = ref_file.readline()
        ref_line_split = ref_line.split()
        for line in cmp_content:
            line_split = line.split()
            if ref_line_split[0] in line_split:
                print(ref_line_split[0])
                ref_joints_str = ref_line_split[5:]
                cmp_joints_str = line.split()[5:]
                ref_joints_int = list(map(int, ref_joints_str))
                cmp_joints_int = list(map(int, cmp_joints_str))
                ref_joints_vec = np.array(ref_joints_int)
                cmp_joints_vec = np.array(cmp_joints_int)
                dist = FasCPM._distance_cosine(ref_joints_vec, cmp_joints_vec)
                cmp_res.write(ref_line_split[0] + ' ' + str(dist) + "\n")
                print(ref_line_split[0], str(dist))
            else:
                continue
                # cmp_file.close()
                # raise ValueError('{} is not the same with the cmpare file'.format(ref_line[0]))
    cmp_file.close()


if __name__ == '__main__':
    # model_save()

    # main_img()

    # main_imgs()

    # main_video()
    main_imgs_no_label()
    # eval_imgs()
    # dataset_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/Eval'
    # gt_file = os.path.join(dataset_dir, 'labels.txt')
    # new_labels_file = os.path.join(dataset_dir, 'labels_by_ssd.txt')
    # cmp_file = os.path.join(dataset_dir, 'result_by_SSD.txt')
    # cmp_result(gt_file, new_labels_file, cmp_file)

    # label_ref, label_cmp, cmp_file = eval_SSD()
    # cmp_result(label_ref, label_cmp, cmp_file)
