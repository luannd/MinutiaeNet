"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""

from __future__ import absolute_import
from __future__ import division

from time import time
from datetime import datetime
from CoarseNet_utils import *
from scipy import misc, ndimage, signal, sparse, io
import scipy.ndimage
import cv2
import sys,os
sys.path.append(os.path.realpath('../FineNet'))
from FineNet_model import FineNetmodel

from keras.models import Model
from keras.layers import Input
from keras import layers
from keras.layers.core import Flatten,Activation,Lambda, Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D,UpSampling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model


import tensorflow as tf

from MinutiaeNet_utils import *
from LossFunctions import *


def conv_bn(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same',
        strides=strides,
        dilation_rate=dilation_rate,
        name='conv-'+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    return top

def conv_bn_prelu(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    if dilation_rate == (1,1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'

    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same',
        strides=strides,
        dilation_rate=dilation_rate,
        name=conv_type+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    top = PReLU(alpha_initializer='zero', shared_axes=[1,2], name='prelu-'+name)(top)
    # top = Dropout(0.25)(top)
    return top

def CoarseNetmodel(input_shape=(400,400,1), weights_path=None, mode='train'):
    # Change network architecture here!!
    img_input=Input(input_shape)
    bn_img=Lambda(img_normalization, name='img_normalized')(img_input)

    # Main part
    conv = conv_bn_prelu(bn_img, (64, 5, 5), '1_0')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_1')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_2')
    conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    # =======Block 1 ========
    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1b')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2b')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3b')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1c')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2c')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3c')
    conv = layers.add([conv, conv1])

    conv_block1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================

    # =======Block 2 ========
    conv1 = conv_bn_prelu(conv_block1, (256,3,3), '3_1')
    conv = conv_bn_prelu(conv1, (256,3,3), '3_2')
    conv = conv_bn_prelu(conv, (256,3,3), '3_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (256, 3, 3), '3_1b')
    conv = conv_bn_prelu(conv1, (256, 3, 3), '3_2b')
    conv = conv_bn_prelu(conv, (256, 3, 3), '3_3b')
    conv = layers.add([conv, conv1])

    conv_block2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================

    # =======Block 3 ========
    conv1 = conv_bn_prelu(conv_block2, (512, 3, 3), '3_1c')
    conv = conv_bn_prelu(conv1, (512, 3, 3), '3_2c')
    conv = conv_bn_prelu(conv, (512, 3, 3), '3_3c')
    conv = layers.add([conv, conv1])
    conv_block3 = conv_bn_prelu(conv, (256, 3, 3), '3_4c')

    #conv_block3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================


    # multi-scale ASPP
    level_2=conv_bn_prelu(conv_block3, (256,3,3), '4_1', dilation_rate=(1,1))
    ori_1=conv_bn_prelu(level_2, (128,1,1), 'ori_1_1')
    ori_1=Conv2D(90, (1,1), padding='same', name='ori_1_2')(ori_1)
    seg_1=conv_bn_prelu(level_2, (128,1,1), 'seg_1_1')
    seg_1=Conv2D(1, (1,1), padding='same', name='seg_1_2')(seg_1)

    level_3=conv_bn_prelu(conv_block2, (256,3,3), '4_2', dilation_rate=(4,4))
    ori_2=conv_bn_prelu(level_3, (128,1,1), 'ori_2_1')
    ori_2=Conv2D(90, (1,1), padding='same', name='ori_2_2')(ori_2)
    seg_2=conv_bn_prelu(level_3, (128,1,1), 'seg_2_1')
    seg_2=Conv2D(1, (1,1), padding='same', name='seg_2_2')(seg_2)

    level_4=conv_bn_prelu(conv_block2, (256,3,3), '4_3', dilation_rate=(8,8))
    ori_3=conv_bn_prelu(level_4, (128,1,1), 'ori_3_1')
    ori_3=Conv2D(90, (1,1), padding='same', name='ori_3_2')(ori_3)
    seg_3=conv_bn_prelu(level_4, (128,1,1), 'seg_3_1')
    seg_3=Conv2D(1, (1,1), padding='same', name='seg_3_2')(seg_3)

    # sum fusion for ori
    ori_out=Lambda(merge_sum)([ori_1, ori_2, ori_3])
    ori_out_1=Activation('sigmoid', name='ori_out_1')(ori_out)
    ori_out_2=Activation('sigmoid', name='ori_out_2')(ori_out)

    # sum fusion for segmentation
    seg_out=Lambda(merge_sum)([seg_1, seg_2, seg_3])
    seg_out=Activation('sigmoid', name='seg_out')(seg_out)

    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)

    filter_img_real = Conv2D(filters_cos.shape[3],(filters_cos.shape[0],filters_cos.shape[1]),
        weights=[filters_cos, np.zeros([filters_cos.shape[3]])], padding='same',
        name='enh_img_real_1')(img_input)
    filter_img_imag = Conv2D(filters_sin.shape[3],(filters_sin.shape[0],filters_sin.shape[1]),
        weights=[filters_sin, np.zeros([filters_sin.shape[3]])], padding='same',
        name='enh_img_imag_1')(img_input)

    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak) # select max ori and set it to 1

    # Use this function to upsample image
    upsample_ori = UpSampling2D(size=(8,8))(ori_peak)
    seg_round = Activation('softsign')(seg_out)


    upsample_seg = UpSampling2D(size=(8,8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])

    enh_img_real = Lambda(reduce_sum, name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])

    enh_img_imag = Lambda(reduce_sum, name='enh_img_imag_2')(mul_mask_imag)
    enh_img = Lambda(atan2, name='phase_img')([enh_img_imag, enh_img_real])

    enh_seg_img = Lambda(merge_concat, name='phase_seg_img')([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    # =======Block 1 ========
    mnt_conv1 = conv_bn_prelu(enh_seg_img, (64, 9, 9), 'mnt_1_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 2 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 3 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (256, 3, 3), 'mnt_3_1')
    mnt_conv2 = conv_bn_prelu(mnt_conv1, (256, 3, 3), 'mnt_3_2')
    mnt_conv3 = conv_bn_prelu(mnt_conv2, (256, 3, 3), 'mnt_3_3')
    mnt_conv3 = layers.add([mnt_conv3, mnt_conv1])
    mnt_conv4 = conv_bn_prelu(mnt_conv3, (256, 3, 3), 'mnt_3_4')
    mnt_conv4 = layers.add([mnt_conv4, mnt_conv2])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv4)
    # ==========================


    mnt_o_1=Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2=conv_bn_prelu(mnt_o_1, (256,1,1), 'mnt_o_1_1')
    mnt_o_3=Conv2D(180, (1,1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out=Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_w_1_1')
    mnt_w_2=Conv2D(8, (1,1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out=Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_h_1_1')
    mnt_h_2=Conv2D(8, (1,1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out=Activation('sigmoid', name='mnt_h_out')(mnt_h_2)

    mnt_s_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_s_1_1')
    mnt_s_2=Conv2D(1, (1,1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out=Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

    if mode == 'deploy':
        model = Model(inputs=[img_input,], outputs=[enh_img, enh_img_imag, enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])
    else:
        model = Model(inputs=[img_input,], outputs=[ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])

    if weights_path != None:
        model.load_weights(weights_path, by_name=True)
    return model

def train(input_shape=(400,400), train_set = None,output_dir='../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S'),
          pretrain_dir=None,batch_size=1,test_set=None, learning_config=None, logging=None):

    img_name, folder_name, img_size = get_maximum_img_size_and_names(train_set, None, max_size=input_shape)

    main_net_model = CoarseNetmodel((img_size[0], img_size[1], 1), pretrain_dir, 'train')
    # Save model architecture
    plot_model(main_net_model, to_file=output_dir+'/model.png',show_shapes=True)

    main_net_model.compile(optimizer=learning_config,
                           loss={'seg_out':segmentation_loss,
                                 'mnt_o_out': orientation_output_loss, 'mnt_w_out': orientation_output_loss,
                                 'mnt_h_out': orientation_output_loss, 'mnt_s_out': minutiae_score_loss
                                 },
                           loss_weights={'seg_out': .5, 'mnt_w_out': .5, 'mnt_h_out': .5, 'mnt_o_out': 100., 'mnt_s_out': 50.},
                           metrics={'seg_out':[seg_acc_pos, seg_acc_neg, seg_acc_all],
                                    'mnt_o_out': [mnt_acc_delta_10, ],
                                    'mnt_w_out': [mnt_mean_delta, ],
                                    'mnt_h_out': [mnt_mean_delta, ],
                                    'mnt_s_out': [seg_acc_pos, seg_acc_neg, seg_acc_all]})

    writer = tf.summary.FileWriter(output_dir)

    Best_F1_result = 0
    Best_loss = 10000000
    for epoch in range(1000):
        outdir = "%s/saved_best_loss/" % (output_dir)
        mkdir(outdir)

        for i, train in enumerate(load_data((img_name, folder_name, img_size), tra_ori_model, rand=True, aug=0.7, batch_size=batch_size)):
            loss = main_net_model.train_on_batch(train[0],
                                                 {'seg_out': train[3],
                                                  'mnt_w_out': train[4], 'mnt_h_out': train[5], 'mnt_o_out': train[6],
                                                  'mnt_s_out': train[7]
                                                  })
            # Save the lowest loss for easy converge
            if Best_loss > loss[0]:
                savedir = "%s%s_%d_%s" % (outdir, str(epoch),i,str(loss[0]))
                main_net_model.save_weights(savedir, True)
                Best_loss = loss[0]

            # Write log on screen at every 20 epochs
            if i%(2/batch_size) == 0:
                logging.info("epoch=%d, step=%d", epoch, i)
                # Write details loss
                logging.info("%s", " ".join(["%s:%.4f\t"%(x) for x in zip(main_net_model.metrics_names, loss)]))
                # logging.info("Loss = %f    Best loss = %f",loss[0],Best_loss)

                # Show in tensorboard
                for name, value in zip(main_net_model.metrics_names, loss):
                    summary = tf.Summary(value=[tf.Summary.Value(tag=name,simple_value=value), ])
                    writer.add_summary(summary, i)

        # Evaluate every 5 epoch: for faster training
        if epoch%10 == 0:
            outdir = "%s/saved_models/" % (output_dir)
            mkdir(outdir)
            savedir = "%s%s" % (outdir, str(epoch))
            main_net_model.save_weights(savedir, True)

            for folder in test_set:
                precision_test, recall_test, F1_test, precision_test_location, recall_test_location, F1_test_location = evaluate_training(savedir, [folder, ], logging=logging)

            summary = tf.Summary(value=[tf.Summary.Value(tag="Precision", simple_value=precision_test),
                                        tf.Summary.Value(tag="Recall", simple_value=recall_test),
                                        tf.Summary.Value(tag="F1", simple_value=F1_test),
                                        tf.Summary.Value(tag="Location Precision", simple_value=precision_test_location),
                                        tf.Summary.Value(tag="Location Recall", simple_value=recall_test_location),
                                        tf.Summary.Value(tag="Location F1", simple_value=F1_test_location), ])
            writer.add_summary(summary, epoch)

        # Only save the best result
        if F1_test > Best_F1_result:
            Best_F1_result = F1_test
        # else:
        #     os.remove(savedir)

    writer.close()
    return


def evaluate_training(model_dir, test_set, logging=None, FineNet_path=None):
    logging.info("Evaluating %s:" % (test_set))

    # Prepare input info
    img_name, folder_name, img_size = get_maximum_img_size_and_names(test_set)


    main_net_model = CoarseNetmodel((None, None, 1), model_dir, 'test')

    ave_prf_nms,ave_prf_nms_location = [],[]

    for j, test in enumerate(
            load_data((img_name, folder_name, img_size), tra_ori_model, rand=False, aug=0.0, batch_size=1)):

        # logging.info("%d / %d: %s"%(j+1, len(img_name), img_name[j]))
        ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(test[0])
        mnt_gt = label2mnt(test[7], test[4], test[5], test[6])

        original_image = test[0].copy()

        mnt_s_out = mnt_s_out * seg_out

        # Does not useful to use this while training
        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05
        isHavingFineNet = False

        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=early_minutiae_thres)
            # Previous exp: 0.2
            mnt_nms_1 = py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []
        if isHavingFineNet == True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if FineNet_path != None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                     y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                        ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_FineNet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [isMinutiaeProb] = model_FineNet.predict(patch_minu)
                        isMinutiaeProb = isMinutiaeProb[0]
                        # print isMinutiaeProb
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4 * tmp_mnt[3] + isMinutiaeProb) / 5
                        mnt_refined.append(tmp_mnt)

                    except:
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]

        p, r, f, l, o = metric_P_R_F(mnt_gt, mnt_nms, 16, np.pi/6)
        ave_prf_nms.append([p, r, f, l, o])
        p, r, f, l, o = metric_P_R_F(mnt_gt, mnt_nms, 16, np.pi)
        ave_prf_nms_location.append([p, r, f, l, o])


    logging.info("Average testing results:")
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    ave_prf_nms_location = np.mean(np.array(ave_prf_nms_location), 0)
    logging.info(
        "Precision: %f\tRecall: %f\tF1-measure: %f\tLocation_dis: %f\tOrientation_delta:%f\n----------------\n" % (
            ave_prf_nms[0], ave_prf_nms[1], ave_prf_nms[2], ave_prf_nms[3], ave_prf_nms[4]))

    return ave_prf_nms[0], ave_prf_nms[1], ave_prf_nms[2], ave_prf_nms_location[0], ave_prf_nms_location[1], ave_prf_nms_location[2]

def fuse_minu_orientation(dir_map, mnt, mode=1,block_size=16):
    # mode is the way to fuse output minutiae with orientation
    # 1: use orientation; 2: use minutiae; 3: fuse average
    blkH, blkW = dir_map.shape
    dir_map = dir_map%(2*np.pi)

    if mode == 1:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation
            ori_value = dir_map[int(mnt[k, 1]//block_size),int(mnt[k, 0]//block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi*2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if np.pi/2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3*np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if 3*np.pi/2 < mnt[k, 2] and mnt[k, 2] <= 2*np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (mnt[k, 2] - ori_value) < (np.pi*2 - mnt[k, 2] + ori_value - np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value


    elif mode == 2:
        return
    elif mode ==3:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation

            ori_value = dir_map[int(mnt[k, 1] // block_size), int(mnt[k, 0] // block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi * 2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3 * np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if 3 * np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= 2 * np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi * 2 - mnt[k, 2] + ori_value - np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value

            mnt[k, 2] = (mnt[k, 2] + fixed_ori)/2.0
    else:
        return

def deploy_with_GT(deploy_set, output_dir, model_path, FineNet_path=None, set_name=None):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]

    # Read image and GT
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    mkdir(output_dir + '/'+ set_name + '/')
    mkdir(output_dir + '/' + set_name + '/mnt_results/')
    mkdir(output_dir + '/'+ set_name + '/seg_results/')
    mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting %s:" % (set_name))

    isHavingFineNet = False

    main_net_model = CoarseNetmodel((None, None, 1), model_path, mode='deploy')

    if isHavingFineNet == True:
        # ====== Load FineNet to verify
        model_FineNet = FineNetmodel(num_classes=2,
                             pretrained_path=FineNet_path,
                             input_shape=(224,224,3))

        model_FineNet.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0),
                      metrics=['accuracy'])

    time_c = []
    ave_prf_nms=[]
    for i, test in enumerate(
            load_data((img_name, folder_name, img_size), tra_ori_model, rand=False, aug=0.0, batch_size=1)):

        print i, img_name[i]
        logging.info("%s %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))
        time_start = time()

        image = misc.imread(deploy_set + 'img_files/' + img_name[i] + '.bmp', mode='L')# / 255.0
        mask = misc.imread(deploy_set + 'seg_files/' + img_name[i] + '.bmp', mode='L') / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]
        mask = mask[:img_size[0], :img_size[1]]


        original_image = image.copy()


        # Generate OF
        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)

        time_afterconv = time()

        # Use post processing to smooth image
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # If use mask from outside
        # seg_out = cv2.resize(mask, dsize=(seg_out.shape[1], seg_out.shape[0]))

        mnt_gt = label2mnt(test[7], test[4], test[5], test[6])

        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05


        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out,
                            thresh=early_minutiae_thres)

            # Previous exp: 0.2
            mnt_nms_1 = py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05


        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []
        if isHavingFineNet == True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if FineNet_path != None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                     y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                        ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_FineNet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [isMinutiaeProb] = model_FineNet.predict(patch_minu)
                        isMinutiaeProb = isMinutiaeProb[0]
                        # print isMinutiaeProb
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4*tmp_mnt[3] + isMinutiaeProb) / 5
                        mnt_refined.append(tmp_mnt)

                    except:
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]

        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)

        # Show the orientation
        show_orientation_field(original_image, dir_map + np.pi, mask=final_mask,
                               fname="%s/%s/OF_results/%s_OF.jpg" % (output_dir, set_name, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/mnt_results/%s.mnt" % (output_dir, set_name, img_name[i]))
        draw_minutiae_overlay_with_score(image, mnt_nms, mnt_gt[:, :3], "%s/%s/%s_minu.jpg"%(output_dir, set_name, img_name[i]),saveimage=True)
        # misc.imsave("%s/%s/%s_score.jpg"%(output_dir, set_name, img_name[i]), np.squeeze(mnt_s_out_upscale))

        misc.imsave("%s/%s/seg_results/%s_seg.jpg" % (output_dir, set_name, img_name[i]), final_mask)

        time_afterdraw = time()
        time_c.append([time_afterconv - time_start, time_afterpost - time_afterconv, time_afterdraw - time_afterpost])
        logging.info(
            "load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f" % (time_c[-1][0], time_c[-1][1], time_c[-1][2]))

        # Metrics calculating
        p, r, f, l, o = metric_P_R_F(mnt_gt, mnt_nms)
        ave_prf_nms.append([p, r, f, l, o])
        print p,r,f

    time_c = np.mean(np.array(time_c), axis=0)
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    print "Precision: %f\tRecall: %f\tF1-measure: %f" % (ave_prf_nms[0], ave_prf_nms[1], ave_prf_nms[2])

    logging.info(
        "Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f" % (time_c[0], time_c[1], time_c[2]))
    return

def inference(deploy_set, output_dir, model_path, FineNet_path=None, set_name=None, file_ext='.bmp', isHavingFineNet = False):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]


    mkdir(output_dir + '/'+ set_name + '/')
    mkdir(output_dir + '/' + set_name + '/mnt_results/')
    mkdir(output_dir + '/'+ set_name + '/seg_results/')
    mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting %s:" % (set_name))

    _, img_name = get_files_in_folder(deploy_set+ 'img_files/', file_ext)
    print deploy_set

    # ====== Load FineNet to verify
    if isHavingFineNet == True:
        model_FineNet = FineNetmodel(num_classes=2,
                             pretrained_path=FineNet_path,
                             input_shape=(224,224,3))

        model_FineNet.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0),
                      metrics=['accuracy'])

    time_c = []

    main_net_model = CoarseNetmodel((None, None, 1), model_path, mode='deploy')

    for i in xrange(0, len(img_name)):
        print i

        image = misc.imread(deploy_set + 'img_files/'+ img_name[i] + file_ext, mode='L')  # / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8

        # read the mask from files
        try:
            mask = misc.imread(deploy_set + 'seg_files/' + img_name[i] + '.jpg', mode='L') / 255.0
        except:
            mask = np.ones((img_size[0],img_size[1]))


        image = image[:img_size[0], :img_size[1]]
        mask = mask[:img_size[0], :img_size[1]]

        original_image = image.copy()

        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

        image = image*mask

        logging.info("%s %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))
        time_start = time()

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)
        time_afterconv = time()

        # If use mask from model
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # If use mask from outside
        # seg_out = cv2.resize(mask, dsize=(seg_out.shape[1], seg_out.shape[0]))


        max_num_minu = 20
        min_num_minu = 6

        early_minutiae_thres = 0.5


        # New adaptive threshold
        mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out,
                        thresh=0)

        # Previous exp: 0.2
        mnt_nms_1 = py_cpu_nms(mnt, 0.5)
        mnt_nms_2 = nms(mnt)
        mnt_nms_1.view('f8,f8,f8,f8').sort(order=['f3'], axis=0)
        mnt_nms_1 = mnt_nms_1[::-1]

        mnt_nms_1_copy = mnt_nms_1.copy()
        mnt_nms_2_copy = mnt_nms_2.copy()
        # Adaptive threshold goes here
        # Make sure the maximum number of minutiae is max_num_minu

        # Sort minutiae by score
        while early_minutiae_thres > 0:
            mnt_nms_1 = mnt_nms_1_copy[mnt_nms_1_copy[:, 3] > early_minutiae_thres, :]
            mnt_nms_2 = mnt_nms_2_copy[mnt_nms_2_copy[:, 3] > early_minutiae_thres, :]

            if mnt_nms_1.shape[0]>max_num_minu or mnt_nms_2.shape[0]>max_num_minu:
                mnt_nms_1 = mnt_nms_1[:max_num_minu,:]
                mnt_nms_2 = mnt_nms_2[:max_num_minu, :]
            if mnt_nms_1.shape[0] > min_num_minu and mnt_nms_2.shape[0] > min_num_minu:
                break

            early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        final_minutiae_score_threashold = early_minutiae_thres - 0.05

        print early_minutiae_thres, final_minutiae_score_threashold

        mnt_refined = []
        if isHavingFineNet == True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if FineNet_path != None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                                    y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(patch_minu, dsize=(224, 224),interpolation=cv2.INTER_NEAREST)

                        ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_FineNet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])


                        # Use soft decision: merge FineNet score with CoarseNet score
                        [isMinutiaeProb] = model_FineNet.predict(patch_minu)
                        isMinutiaeProb = isMinutiaeProb[0]
                        #print isMinutiaeProb
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4*tmp_mnt[3] + isMinutiaeProb)/5
                        mnt_refined.append(tmp_mnt)

                    except:
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:,3]>final_minutiae_score_threashold,:]

        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)
        # Show the orientation
        show_orientation_field(original_image, dir_map + np.pi, mask=final_mask, fname="%s/%s/OF_results/%s_OF.jpg" % (output_dir, set_name, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/mnt_results/%s.mnt"%(output_dir, set_name, img_name[i]))
        draw_minutiae(original_image, mnt_nms, "%s/%s/%s_minu.jpg"%(output_dir, set_name, img_name[i]),saveimage=True)

        misc.imsave("%s/%s/seg_results/%s_seg.jpg" % (output_dir, set_name, img_name[i]), final_mask)

        time_afterdraw = time()
        time_c.append([time_afterconv - time_start, time_afterpost - time_afterconv, time_afterdraw - time_afterpost])
        logging.info(
            "load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f" % (time_c[-1][0], time_c[-1][1], time_c[-1][2]))


    # time_c = np.mean(np.array(time_c), axis=0)
    # logging.info(
    #     "Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f" % (time_c[0], time_c[1], time_c[2]))
    return