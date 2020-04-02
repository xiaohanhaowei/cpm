#!/home/wanghongwei/anaconda3/envs/tf114/bin/python
import tensorflow as tf
import numpy as np
import cv2
import os
import importlib
import time
import sys

from utils import cpm_utils
from config import FLAGS
# from config import PIN_FLAGS
import Ensemble_data_generator
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
# import the directory 'models/nets/PIN_FLAGS.network_def' module


def main(argv):
    # tf.enable_eager_execution()
    """

    :param argv:
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    """ Create dirs for saving models and logs
    """
    # added by hongwei.wang
    # FLAGS.network_def --> PIN_FLAGS.network_def
    model_path_suffix = os.path.join(FLAGS.network_def,
                                     'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                     'joints_{}'.format(FLAGS.num_of_joints),
                                     'stages_{}'.format(FLAGS.cpm_stages),
                                     'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step)
                                     )
    model_save_dir = os.path.join('./models',
                                  'weights',
                                  model_path_suffix)
    train_log_save_dir = os.path.join('./models',
                                      'logs',
                                      model_path_suffix,
                                      'train')
    test_log_save_dir = os.path.join('./models',
                                     'logs',
                                     model_path_suffix,
                                     'test')
    print(model_save_dir)
    os.system('mkdir -p {}'.format(model_save_dir))
    os.system('mkdir -p {}'.format(train_log_save_dir))
    os.system('mkdir -p {}'.format(test_log_save_dir))

    """ Create data generator
    @hongwei.wang
    TODO  There is no Ensemble_data_generator.Need to rewrite the py file.
    FIXME:
    """
    # g = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
    #                                                     FLAGS.bg_img_dir,
    #                                                     FLAGS.batch_size, FLAGS.input_size, True, True,
    #                                                     FLAGS.augmentation_config, FLAGS.hnm, FLAGS.do_cropping)
    # g_eval = Ensemble_data_generator.ensemble_data_generator(FLAGS.val_img_dir,
    #                                                          FLAGS.bg_img_dir,
    #                                                          FLAGS.batch_size, FLAGS.input_size, True, True,
    #                                                          FLAGS.augmentation_config, FLAGS.hnm, FLAGS.do_cropping)
    g = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
                                                        FLAGS.batch_size, 
                                                        FLAGS.input_size, 
                                                        FLAGS.box_size,
                                                        FLAGS.bg_img_dir,
                                                        augmentation_flag=True,
                                                        augmentation_config=None
                                                        )
    g_eval = Ensemble_data_generator.ensemble_data_generator(FLAGS.val_img_dir,
                                                             FLAGS.batch_size, 
                                                             FLAGS.input_size, 
                                                             FLAGS.box_size,
                                                             FLAGS.bg_img_dir,
                                                             augmentation_flag=True,
                                                             augmentation_config=None
                                                             )
    """ Build network graph
    """
    model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=True)

    model.build_loss(FLAGS.init_lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step, optimizer='RMSProp')
    print('=====Model Build=====\n')

    merged_summary = tf.summary.merge_all()

    """ Training
    """
    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    with tf.Session(config=tf.ConfigProto(device_count=device_count,
                                          allow_soft_placement=True)) as sess:
        # Create tensorboard
        train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Restore pretrained weights
        if FLAGS.pretrained_model != '':
            if FLAGS.pretrained_model.endswith('.pkl'):
                model.load_weights_from_file(FLAGS.pretrained_model, sess, finetune=True)

                # Check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

            else:
                # pre_tr_model = os.path.join(FLAGS.pretrained_model)
                pre_tr_model = FLAGS.pretrained_model
                ckpt = tf.train.get_checkpoint_state(pre_tr_model)
                # assert len(os.listdir(pre_tr_model)) > 0
                # saver.restore(sess, pre_tr_model)
                if ckpt and ckpt.model_checkpoint_path:
                    print("m" * 20, "load from ckpt")
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

        for training_itr in range(FLAGS.training_iters):
            t1 = time.time()

            # Read one batch data

            batch_x, batch_joints = g.next
            batch_orig_x = g.orig_next
            batch_bg_x = g.bg_next
            # print('########' * 10)
            # orig_x_np = sess.run(batch_orig_x)
            # bg_x_np = sess.run(batch_bg_x)
            batch_x_np, batch_joints_np, bg_x_np, orig_x_np = sess.run([batch_x, batch_joints, batch_bg_x, batch_orig_x])
            print(batch_x_np.shape)
            cv2.imshow('backgroud', bg_x_np[0])
            cv2.imshow('origin', orig_x_np[0][0])
            cv2.imshow('mixup', batch_x_np[0])
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
            print(type(batch_x_np))

            if FLAGS.normalize_img:
                # Normalize images
                batch_x_np = batch_x_np / 255.0 - 0.5
            else:
                batch_x_np -= 128.0

            # Generate heatmaps from joints
            batch_gt_heatmap_np = cpm_utils.make_heatmaps_from_joints(FLAGS.input_size,
                                                                      FLAGS.heatmap_size,
                                                                      FLAGS.joint_gaussian_variance,
                                                                      batch_joints_np)

            # Forward and update weights
            stage_losses_np, total_loss_np, _, summaries, current_lr, stage_heatmap_np, global_step = sess.run([model.stage_loss,
                             model.total_loss,
                             model.train_op,
                             merged_summary,
                             model.cur_lr,
                             model.stage_heatmap,
                             model.global_step], 
                             feed_dict={model.input_images: batch_x_np,
                                        model.gt_hmap_placeholder: batch_gt_heatmap_np})

            # Show training info
            print_current_training_stats(global_step, current_lr, stage_losses_np, total_loss_np, time.time() - t1)

            # Write logs
            if (global_step + 1) % 100 == 0:
                train_writer.add_summary(summaries, global_step)

            # Draw intermediate results
            if (global_step + 1) % 10 == 0:
                if FLAGS.color_channel == 'GRAY':
                    demo_img = np.repeat(batch_x_np[0], 3, axis=2)
                    if FLAGS.normalize_img:
                        demo_img += 0.5
                    else:
                        demo_img += 128.0
                        demo_img /= 255.0
                elif FLAGS.color_channel == 'RGB':
                    if FLAGS.normalize_img:
                        demo_img = batch_x_np[0] + 0.5
                    else:
                        demo_img += 128.0
                        demo_img /= 255.0
                else:
                    raise ValueError('Non support image type.')
                demo_stage_heatmaps = []
                for stage in range(FLAGS.cpm_stages):
                    print("stage=", stage)
                    demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape((FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                    demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
                    demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
                    demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                    demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                    demo_stage_heatmaps.append(demo_stage_heatmap)

                demo_gt_heatmap = batch_gt_heatmap_np[0, :, :, 0:FLAGS.num_of_joints].reshape(
                    (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                demo_gt_heatmap = cv2.resize(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size))
                demo_gt_heatmap = np.amax(demo_gt_heatmap, axis=2)
                demo_gt_heatmap = np.reshape(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                demo_gt_heatmap = np.repeat(demo_gt_heatmap, 3, axis=2)

                if FLAGS.cpm_stages > 4:
                    upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]),
                                               axis=1)
                    if FLAGS.normalize_img:
                        blend_img = 0.5 * demo_img + 0.5 * demo_gt_heatmap
                    else:
                        blend_img = 0.5 * demo_img / 255.0 + 0.5 * demo_gt_heatmap
                    lower_img = np.concatenate((demo_stage_heatmaps[FLAGS.cpm_stages - 1], demo_gt_heatmap, blend_img),
                                               axis=1)
                    demo_img = np.concatenate((upper_img, lower_img), axis=0)
                    cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)
                else:
                    upper_img = np.concatenate((demo_stage_heatmaps[FLAGS.cpm_stages - 1], demo_gt_heatmap, demo_img),
                                               axis=1)
                    cv2.imshow('current heatmap', (upper_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)

            if (global_step + 1) % FLAGS.validation_iters == 0:
                mean_val_loss = 0
                cnt = 0
                while cnt < 10:
                    batch_x_np, batch_joints = sess.run(g_eval.next)
                    # Normalize images
                    batch_x_np = batch_x_np / 255.0 - 0.5

                    batch_gt_heatmap_np = cpm_utils.make_heatmaps_from_joints(FLAGS.input_size,
                                                                              FLAGS.heatmap_size,
                                                                              FLAGS.joint_gaussian_variance,
                                                                              batch_joints_np)
                    total_loss_np, summaries = sess.run([model.total_loss, merged_summary],
                                                        feed_dict={model.input_images: batch_x_np,
                                                                   model.gt_hmap_placeholder: batch_gt_heatmap_np})
                    mean_val_loss += total_loss_np
                    cnt += 1

                print('\nValidation loss: {:>7.2f}\n'.format(mean_val_loss / cnt))
                test_writer.add_summary(summaries, global_step)

            # Save models
            if (global_step + 1) % FLAGS.model_save_iters == 0:
                print("ab" * 22)
                print('********' * 10)
                print(model_save_dir)
                saver.save(sess=sess, save_path=model_save_dir,
                           global_step=(global_step + 1))
                print('\nModel checkpoint saved...\n')
            # Finish training
            if global_step == FLAGS.training_iters:
                break
    print('Training done.')


def print_current_training_stats(global_step, cur_lr, stage_losses, total_loss, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGS.training_iters, cur_lr, time_elapsed)
    losses = ' | '.join(
        ['S{} loss: {:>7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in range(FLAGS.cpm_stages)])
    losses += ' | Total loss: {}'.format(total_loss)
    print(stats)
    print(losses + '\n')


if __name__ == '__main__':
    tf.app.run()
