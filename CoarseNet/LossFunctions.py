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

from keras.models import Model
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, GlobalAveragePooling2D
from keras.layers import Input, Lambda, MaxPooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
import numpy as np
from CoarseNet_utils import *

def orientation_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32)
    # weighted cross entropy loss
    lamb_pos, lamb_neg = 1., 1.
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())

    # coherence loss, nearby ori should be as near as possible
    # Oritentation coherence loss

    # 3x3 ones kernel
    mean_kernal = np.reshape(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/8, [3, 3, 1, 1])

    sin2angle_ori, cos2angle_ori, modulus_ori = ori2angle(y_pred)
    sin2angle = K.conv2d(sin2angle_ori, mean_kernal, padding='same')
    cos2angle = K.conv2d(cos2angle_ori, mean_kernal, padding='same')
    modulus = K.conv2d(modulus_ori, mean_kernal, padding='same')

    coherence = K.sqrt(K.square(sin2angle) + K.square(cos2angle)) / (modulus + K.epsilon())
    coherenceloss = K.sum(label_seg) / (K.sum(coherence*label_seg) + K.epsilon()) - 1
    loss = logloss + lamb*coherenceloss
    return loss

def orientation_output_loss(y_true, y_pred):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32)
    # weighted cross entropy loss
    lamb_pos, lamb_neg= 1., 1.
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())
    return logloss

def segmentation_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # weighted cross entropy loss
    total_elements = K.sum(K.tf.ones_like(y_true))
    label_pos = K.tf.cast(K.tf.greater(y_true, 0.0), K.tf.float32)
    lamb_pos = 0.5 * total_elements / K.sum(label_pos)
    lamb_neg = 1 / (2 - 1/lamb_pos)
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = -K.mean(K.sum(logloss, axis=-1))
    # smooth loss
    smooth_kernal = np.reshape(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)/8, [3, 3, 1, 1])
    smoothloss = K.mean(K.abs(K.conv2d(y_pred, smooth_kernal)))
    loss = logloss + lamb*smoothloss
    return loss

def minutiae_score_loss(y_true, y_pred):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.tf.cast(K.tf.not_equal(y_true, 0.0), K.tf.float32)
    y_true = K.tf.where(K.tf.less(y_true,0.0), K.tf.zeros_like(y_true), y_true) # set -1 -> 0
    # weighted cross entropy loss
    total_elements = K.sum(label_seg) + K.epsilon()
    lamb_pos, lamb_neg = 10., .5
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    # apply ROI
    logloss = logloss*label_seg
    logloss = -K.sum(logloss) / total_elements
    return logloss