import os
from generator import *
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.applications import InceptionV3, ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy


def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 300:
        lr *= 0.5e-3
    elif epoch > 220:
        lr *= 1e-3
    elif epoch > 180:
        lr *= 1e-2
    elif epoch > 120:
        lr *= 1e-1
    print 'Learning rate:', lr
    return lr


def get_model(MODEL, width, height):
    base_model = MODEL(weights='imagenet', include_top=False, input_shape=(width, height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(8, activation='sigmoid')(x)

    for layer in base_model.layers:
        layer.trainable = True

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train(epochs, batch_size, width, height, prefix='_01',save_dir='models_and_logs/'):
    model = get_model(InceptionV3, width, height)
#    model = get_model(ResNet50, width, height)

    model.compile(optimizer=SGD(lr=lr_schedule(0)), loss=['binary_crossentropy'], metrics=['accuracy'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    csvlogger = CSVLogger(save_dir + 'log' + str(epochs) + str(batch_size) + prefix + '.log', append=True)
    model_check = ModelCheckpoint(save_dir + 'm' + str(epochs) + str(batch_size) + prefix + '_p.h5', monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    tblogger = TensorBoard(log_dir=save_dir + 'tblogger'+ prefix, histogram_freq=5, write_graph=True, write_images=True)

    model.fit_generator(data_generator(train_dirs, width, height, batch_size),
                        steps_per_epoch=100,
                        epochs=epochs,
                        validation_data=data_generator(valid_dirs, width, height, batch_size),
                        validation_steps=50,
                        verbose=1,
                        workers=100,
                        max_q_size=128,
                        callbacks=[csvlogger, model_check, lr_scheduler, tblogger])

    model.save_weights(save_dir + 'weight' + str(epochs) + str(batch_size) + prefix + '.h5')


train(200, 32, 299, 299, prefix='_py27_01')

