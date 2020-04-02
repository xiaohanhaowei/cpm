import tensorflow as tf
import numpy as np
import cv2
import os
import importlib
import time

from utils import cpm_utils
from config import FLAGS
# from config import PIN_FLAGS
import Ensemble_data_generator


def main():
    tf.enable_eager_execution()
    g = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
                                                        FLAGS.batch_size, 
                                                        FLAGS.input_size, 
                                                        FLAGS.box_size
                                                        )
    # g_eval = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
    #                                                          FLAGS.batch_size, 
    #                                                          FLAGS.input_size, 
    #                                                          FLAGS.box_size
    #                                                          )
    image, joints = g.next()
    print(image)
    print(joints)


if __name__ == '__main__':
    main()
