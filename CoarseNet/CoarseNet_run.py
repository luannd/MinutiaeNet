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
from keras import backend as K

from MinutiaeNet_utils import *
from CoarseNet_utils import *
from CoarseNet_model import *



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

# mode = 'inference'
mode = 'deploy'

# Can use multiple folders for deploy, inference
deploy_set = ['../Dataset/CoarseNet_train/',]
inference_set = ['../Dataset/CoarseNet_test/',]


pretrain_dir = '../Models/CoarseNet.h5'
output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

FineNet_dir = '../Models/FineNet.h5'

def main():
    if mode == 'deploy':
        output_dir = '../output_CoarseNet/deployResults/' +datetime.now().strftime('%Y%m%d-%H%M%S')
        logging = init_log(output_dir)
        for i, folder in enumerate(deploy_set):
            deploy_with_GT(folder, output_dir=output_dir, model_path=pretrain_dir, FineNet_path=FineNet_dir)
            # evaluate_training(model_dir=pretrain_dir, test_set=folder, logging=logging)
    elif mode == 'inference':
        output_dir = '../output_CoarseNet/inferenceResults/' +datetime.now().strftime('%Y%m%d-%H%M%S')
        logging = init_log(output_dir)
        for i, folder in enumerate(inference_set):
            inference(folder, output_dir=output_dir, model_path=pretrain_dir, FineNet_path=FineNet_dir, file_ext='.bmp',
                      isHavingFineNet=False)
    else:
        pass

if __name__ =='__main__':
    main()
