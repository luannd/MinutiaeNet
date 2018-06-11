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

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from datetime import datetime
from MinutiaeNet_utils import *

from keras import backend as K
from keras.optimizers import SGD, Adam

from CoarseNet_utils import *
from CoarseNet_model import *
import argparse

parser = argparse.ArgumentParser(description='Minutiae Net')
parser.add_argument('lr', type=str, default="0.005",
                    help='Setting learning rate')

parser.add_argument('GPU', type=str, default="0",
                    help='Choosing GPU')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

batch_size = 2
use_multiprocessing = False
input_size = 400

# Can use multiple folders for training
train_set = ['../Dataset/CoarseNet_train/',]

validate_set = ['../path/to/your/data/',]

pretrain_dir = '../Models/CoarseNet.h5'
output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')
FineNet_dir = '../Models/FineNet.h5'

if __name__ =='__main__':

    output_dir = '../output_CoarseNet/trainResults/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    logging = init_log(output_dir)
    logging.info("Learning rate = %s", args.lr)
    logging.info("Pretrain dir = %s", pretrain_dir)

    train(input_shape=(input_size, input_size), train_set=train_set, output_dir=output_dir,
          pretrain_dir=pretrain_dir, batch_size=batch_size, test_set=validate_set,
          learning_config=Adam(lr=float(args.lr), beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0.9),
          logging=logging)