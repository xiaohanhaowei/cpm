import tensorflow as tf
from config import FLAGS
# import multiprocessing as mp


class ensemble_data_generator:
    def __init__(self, img_dir, batch_size, image_size, box_size, bgimg_dir=None, augmentation_flag=False, augmentation_config=None):
        # super(mp.Process, self).__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.box_size = box_size
        self.dataset = tf.data.TFRecordDataset([self.img_dir])
        # added by hongwei.wang
        # add the augmentation processing method
        self.bg_dataset = None
        if augmentation_flag:
            if bgimg_dir is None:
                raise IOError("No backgroud image input!")
            else:
                self.bg_dataset = tf.data.TFRecordDataset([bgimg_dir])
        self.dataset = self.dataset.map(lambda x: self._parser(x, box_size=self.box_size, image_size=self.image_size))
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.batch_size).shuffle(100)
        self.iterator = self.dataset.make_one_shot_iterator()
        next_ele = self.iterator.get_next()
        self.orig_next = next_ele
        if augmentation_flag:
            self.bg_dataset = self.bg_dataset.map(lambda x: self._bgparser(x, box_size=self.box_size))
            self.bg_dataset = self.bg_dataset.repeat()
            self.bg_dataset = self.bg_dataset.batch(self.batch_size).shuffle(100)
            self.bg_iterator = self.bg_dataset.make_one_shot_iterator()
            bg_next_ele = self.bg_iterator.get_next()
            self.bg_next = bg_next_ele
            # bg_next_ele = tf.cast(bg_next_ele, tf.float32)
            next_ele = list(next_ele)
            next_ele[0] = tf.add(tf.multiply(tf.cast(bg_next_ele[0], tf.float32), tf.constant(0.08)), tf.multiply(tf.cast(next_ele[0], tf.float32), tf.constant(0.92)))
            next_ele[0] = tf.cast(next_ele[0], tf.uint8)
        self.next = next_ele

    @staticmethod
    def _parser(record, box_size, image_size):
        keys_to_features = {
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'joint': tf.VarLenFeature(dtype=tf.float32)}
            # 'heatmaps': tf.FixedLenFeature([], dtype=tf.float64)}

        # print('[_parser | %f] record:' % time.time(), record.dtype, record.shape)
        # record = tf.cast(record, dtype=tf.string, name='asdfdsfs')
        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [box_size, box_size, 3])
        # @hongwei.wang 
        # TODO: backgroud image mixing_up
        image = tf.image.resize_images(image, [image_size, image_size])

        joints = tf.sparse_tensor_to_dense(parsed['joint'])
        joints = tf.reshape(joints, [FLAGS.num_of_joints, 2], name='ll')  # XXX: name!
        # TODO: Is it necessary to change the follow python logic to tf logic 
        joints *= image_size / box_size  # arbitrary python logic

        # def joints_resize(per_joints):
        #     per_joints *= self.image_size / self.box_size
        # [joints, ] = tf.py_function(joints_resize, [joints], [tf.float32])
        image = tf.cast(image, tf.float32)
        # bright_ness = tf.image.random_brightness(image, max_delta=63)
        # image = tf.image.random_brightness(image, max_delta=1.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.cast(image, tf.uint8)
        return image, joints

    @staticmethod
    def _bgparser(record, box_size):
        keys_to_features = {
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'shape': tf.VarLenFeature(dtype=tf.int64)}
        parsed = tf.parse_single_example(record, keys_to_features)
        bg_img = tf.decode_raw(parsed['image'], tf.uint8)
        # shape = tf.sparse_tensor_to_dense(parsed['shape'])
        bg_img = tf.reshape(bg_img, [box_size, box_size, 3])

        bg_img = tf.image.resize_images(bg_img, [box_size, box_size])
        return bg_img


"""
    def next(self):
        # filenames = [self.img_dir]
        # dataset = tf.data.TFRecordDataset(filenames)
        # import pdb
        # pdb.set_trace()
        self.dataset = self.dataset.map(self._parser)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()
        next_ele = self.iterator.get_next()

        return next_ele
#         def _parser(record):
#             keys_to_features = {
#                 'image': tf.FixedLenFeature([], dtype=tf.string),
#                 'joint': tf.VarLenFeature(dtype=tf.float32)}

#             parsed = tf.parse_single_example(record, keys_to_features)

#             image = tf.decode_raw(parsed['image'], tf.uint8)
#             image = tf.reshape(image, [self.box_size, self.box_size, 3])
#             image = tf.image.resize_images(image, [self.image_size, self.image_size])

#             joints = tf.sparse_tensor_to_dense(parsed['joint'])
#             joints = tf.reshape(joints, [21, 2])
#             #it need to be equally resize as the image
#             joints *= image_size / box_size

#             return image, joints
#         filenames = [self.img_dir]
#         dataset = tf.data.TFRecordDataset(filenames)

#         # Use `Dataset.map()` to build a pair of a feature dictionary and a label
#         # tensor for each example.
#         dataset = dataset.map(_parser)
#         dataset = dataset.repeat()
#         dataset = dataset.batch(self.batch_size)
#         self.iterator = dataset.make_one_shot_iterator()
"""
#     def next(self):
#         # `features` is a dictionary in which each value is a batch of values for
#         # that feature; `labels` is a batch of labels.
#         image, joints = self.iterator.get_next()

# #         with tf.Session() as sess:
# #             image = sess.run(image)
# #             joints = sess.run(joints)

#         return image, joints
