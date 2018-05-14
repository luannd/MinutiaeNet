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

import sys,os
sys.path.append(os.path.realpath('../FineNet'))

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from FineNet_model import FineNetmodel, plot_confusion_matrix

import numpy as np
import os
from sklearn.metrics import confusion_matrix
from datetime import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'


output_dir = '../output_FineNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), output_dir)
log_dir = os.path.join(os.getcwd(), output_dir + '/logs')

# Training parameters
batch_size = 32
epochs = 200
num_classes = 2

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model size, patch
model_type = 'patch224batch32'


# =============== DATA loading ========================

train_path = '../Dataset/train/'
test_path = '../Dataset/validate/'

input_shape = (224, 224, 3)

# Using data augmentation technique for training
datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=180,
        # randomly shift images horizontally
        width_shift_range=0.5,
        # randomly shift images vertically
        height_shift_range=0.5,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True)

train_batches = datagen.flow_from_directory(train_path, target_size=(input_shape[0], input_shape[1]), classes=['minu', 'non_minu'], batch_size=batch_size)
# Feed data from directory into batches
test_gen = ImageDataGenerator()
test_batches = test_gen.flow_from_directory(test_path, target_size=(input_shape[0], input_shape[1]), classes=['minu', 'non_minu'], batch_size=batch_size)


# =============== end DATA loading ========================



def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    lr = 0.5e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 5e-2
    elif epoch > 30:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr




#============== Define model ==================

model = FineNetmodel(num_classes = num_classes,
                     pretrained_path = '../Models/FineNet.h5',
                     input_shape=input_shape)

# Save model architecture
#plot_model(model, to_file='./modelFineNet.pdf',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
#model.summary()

#============== End define model ==============


#============== Other stuffs for loging and parameters ==================
model_name = 'FineNet_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

filepath = os.path.join(save_dir, model_name)


# Show in tensorboard
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard]

#============== End other stuffs  ==================

# Begin training
model.fit_generator(train_batches,
                    validation_data=test_batches,
                    epochs=epochs, verbose=1,
                    callbacks=callbacks)



# Plot confusion matrix
score = model.evaluate_generator(test_batches)
print 'Test accuracy:', score[1]
predictions = model.predict_generator(test_batches)
test_labels = test_batches.classes[test_batches.index_array]

cm = confusion_matrix(test_labels, np.argmax(predictions,axis=1))
cm_plot_labels = ['minu','non_minu']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')