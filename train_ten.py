from generator import *
from keras.models import Model
# from keras.optimizers import SGD
from keras.applications import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

epochs = 200
batch_size = 32
width, height = 299, 299


#def binary_cross_entropy_mean(y_true, y_pred):
#    cross_entropy = binary_crossentropy(y_true, y_pred)
#    cross_entropy_mean = K.mean(cross_entropy, axis=-1)
#    return cross_entropy_mean


def multi_acc(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.round(y_pred), K.round(y_true)), tf.float32), axis=-1)


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, height, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='elu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])

csvlogger = CSVLogger('log20032100_01.log', append=True)
model_check = ModelCheckpoint('model20032100_01_p.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(data_generator(train_dirs, width, height, batch_size),
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data_generator(valid_dirs, width, height, batch_size),
                    validation_steps=50,
                    verbose=1,
                    workers=100,
                    max_q_size=128,
                    callbacks=[csvlogger, model_check])
