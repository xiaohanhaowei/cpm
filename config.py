class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 256
    heatmap_size = 32
    cpm_stages = 3
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 6
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0
    box_size = 256  # in compatible with the `create_cpm_tfr_fulljoints.py`

    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    # DEMO_TYPE = 'test_imgs/139.jpg'
    # DEMO_TYPE = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img3/4424.jpg'
    DEMO_TYPE = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/img3/4424.jpg'

    # DEMO_TYPE = 'SINGLE'

    # model_path = 'cpm_hand'
    model_path = 'cpm_hand'
    cam_id = 0

    webcam_height = 480
    webcam_width = 640

    KALMAN_ON = False
    use_kalman = False
    kalman_noise = 0.03
    cmap_radius = 21

    """
    Training settings
    """
    network_def = 'cpm_hand'
    # added by hongwei.wang
    # now train_img_dir and val_img_dir are moved to the ~/WorkSpace/datasets
    # there is no need to modify them because we don't train the model at local machine

    # train_img_dir = '/home/wanghongwei/WorkSpace/detect/cpm-tf/dataset/train/fas_train_dataset.tfrecords'
    # val_img_dir = '/home/wanghongwei/WorkSpace/detect/cpm-tf/dataset/eval/fas_eval_dataset.tfrecords'    
    train_img_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/tfrecords-from-cpm/train/fas_train_dataset.tfrecords'
    val_img_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/tfrecords-from-cpm/eval/fas_eval_dataset.tfrecords'
    bg_img_dir = '/home/wanghongwei/WorkSpace/datasets/fans/datasets/tfrecords-from-cpm/bg/fas_bg_dataset.tfrecords'
    # pretrained_model = '/home/wanghongwei/WorkSpace/detect/cpm-tf/models/weights/cpm_hand/input_256_output_32/joints_6/stages_3/init_0.001_rate_0.5_step_10000'
    pretrained_model = '/home/wanghongwei/WorkSpace/source/detect/cpm-tf/models/weights/cpm_hand/34-test'
    # pretrained_model_file = '/home/wanghongwei/WorkSpace/source/detect/cpm-tf/models/weights/cpm_hand/finetune-0119/init_0.05_rate_0.7_step_10000-300000'
    # pretrained_model_file = '/home/wanghongwei/WorkSpace/source/detect/cpm-tf/models/weights/cpm_hand/finetune-0120/init_0.07_rate_0.5_step_15000-200000'
    pretrained_model_file = '/home/wanghongwei/WorkSpace/weights/cpm/finetune-0305/init_0.071_rate_0.5_step_20000-300000'

    batch_size = 1
    init_lr = 0.001
    lr_decay_rate = 0.5
    lr_decay_step = 10000
    training_iters = 100000
    verbose_iters = 10
    validation_iters = 1000
    model_save_iters = 500
    augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (-10, 10),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (-0.3, 0.5),
                           'rotate_limit': (-90, 90)}
    hnm = True  # Make sure generate hnm files first
    do_cropping = True

    """
    For Freeze graphs
    """
    output_node_names = 'stage_3/mid_conv7/BiasAdd:0'

    """
    For Drawing
    """
    # Default Pose
    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]

    # Limb connections
    # center-> little finger
    # center -> ring finger
    # center -> middle finger
    # center -> forefinger 
    # center -> firstfinger
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],  # which limb connection
             [0, 5]  
             # [5, 6],
             # [6, 7],
             # [7, 8],  # limb connection
             # [0, 9],
             # [9, 10],
             # [10, 11],
             # [11, 12],  # limb connection
             # [0, 13],
             # [13, 14],
             # [14, 15],
             # [15, 16],  # limb connection
             # [0, 17],
             # [17, 18],
             # [18, 19],
             # [19, 20]  # limb connection
             ]

    # Finger colors
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    # My hand joint order
    # FLAGS.limbs = [[0, 1],
    #          [1, 2],
    #          [2, 3],
    #          [3, 20],
    #          [4, 5],
    #          [5, 6],
    #          [6, 7],
    #          [7, 20],
    #          [8, 9],
    #          [9, 10],
    #          [10, 11],
    #          [11, 20],
    #          [12, 13],
    #          [13, 14],
    #          [14, 15],
    #          [15, 20],
    #          [16, 17],
    #          [17, 18],
    #          [18, 19],
    #          [19, 20]
    #          ]


class PIN_FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 256
    heatmap_size = 32
    cpm_stages = 3
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 6
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0

    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    DEMO_TYPE = 'MULTI'

    model_path = 'cpm_hand'
    cam_id = 0

    webcam_height = 480
    webcam_width = 640
    KALMAN_ON = False
    use_kalman = True
    kalman_noise = 0.03

    """
    Training settings
    """
    network_def = 'cpm_hand'
    train_img_dir = ''
    val_img_dir = ''
    bg_img_dir = ''
    pretrained_model = 'cpm_hand'
    batch_size = 5
    init_lr = 0.001
    lr_decay_rate = 0.5
    lr_decay_step = 10000
    training_iters = 300000
    verbose_iters = 10
    validation_iters = 1000
    model_save_iters = 5000
    augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (-10, 10),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (-0.3, 0.5),
                           'rotate_limit': (-90, 90)}
    hnm = True  # Make sure generate hnm files first
    do_cropping = True

    """
    For Freeze graphs
    """
    output_node_names = 'stage_3/mid_conv7/BiasAdd:0'

    """
    For Drawing
    """
    # Default Pose
    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]

    # Limb connections
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    # Finger colors
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    # My hand joint order
    # FLAGS.limbs = [[0, 1],
    #          [1, 2],
    #          [2, 3],
    #          [3, 20],
    #          [4, 5],
    #          [5, 6],
    #          [6, 7],
    #          [7, 20],
    #          [8, 9],
    #          [9, 10],
    #          [10, 11],
    #          [11, 20],
    #          [12, 13],
    #          [13, 14],
    #          [14, 15],
    #          [15, 20],
    #          [16, 17],
    #          [17, 18],
    #          [18, 19],
    #          [19, 20]
    #          ]
